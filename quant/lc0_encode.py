"""Pure-Python port of lc0 v0.33's input encoder (src/neural/encoder.cc:
EncodePositionForNN with FillEmptyHistory::FEN_ONLY, ChooseTransform,
MoveToNNIndex), so calibration and dev planes can be made on any machine
without lc0's Python bindings. tests/test_encoder.py checks it bit for bit
against the bindings on random games (castling, en passant, repetitions)
and tablebase FENs for every input format.

    from lc0_encode import encode
    planes, legal_idx, legal_uci = encode(fen=None, moves=["e2e4", "e7e5"], input_format=5)
"""
import chess
import numpy as np

try:
    from lc0_moves import INDEX as NN_INDEX
except ImportError:  # imported as quant.lc0_encode
    from .lc0_moves import INDEX as NN_INDEX

FLIP, MIRROR, TRANSPOSE = 1, 2, 4
M64 = (1 << 64) - 1
ALL = M64

CLASSICAL, CASTLING_PLANE, CANON, CANON_HECTO, CANON_HECTO_ARMA, CANON_V2, CANON_V2_ARMA = \
    1, 2, 3, 4, 132, 5, 133


def is_canonical(f):
    return f >= CANON


def is_hecto(f):
    return f >= CANON_HECTO


def is_arma(f):
    return f in (CANON_HECTO_ARMA, CANON_V2_ARMA)


def rev_bits_in_bytes(v):
    v = ((v >> 1) & 0x5555555555555555) | ((v & 0x5555555555555555) << 1)
    v = ((v >> 2) & 0x3333333333333333) | ((v & 0x3333333333333333) << 2)
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0F) | ((v & 0x0F0F0F0F0F0F0F0F) << 4)
    return v & M64


def rev_bytes(v):
    v = ((v & 0x00000000FFFFFFFF) << 32) | ((v & 0xFFFFFFFF00000000) >> 32)
    v = ((v & 0x0000FFFF0000FFFF) << 16) | ((v & 0xFFFF0000FFFF0000) >> 16)
    v = ((v & 0x00FF00FF00FF00FF) << 8) | ((v & 0xFF00FF00FF00FF00) >> 8)
    return v & M64


def transpose_bits(v):
    v = ((v & 0xAA00AA00AA00AA00) >> 9) | ((v & 0x0055005500550055) << 9) | (v & 0x55AA55AA55AA55AA)
    v = ((v & 0xCCCC0000CCCC0000) >> 18) | ((v & 0x0000333300003333) << 18) | (v & 0x3333CCCC3333CCCC)
    v = ((v & 0xF0F0F0F000000000) >> 36) | ((v & 0x000000000F0F0F0F) << 36) | (v & 0x0F0F0F0FF0F0F0F0)
    return v & M64


class LBoard:
    """lc0's ChessBoard: everything from the side-to-move's perspective."""
    __slots__ = ("ours", "theirs", "pawns", "knights", "bishops", "rooks", "queens",
                 "our_king", "their_king", "ep", "cast", "flipped")

    @classmethod
    def from_pychess(cls, b):
        x = cls()
        stm = b.turn
        o = (lambda bb: chess.flip_vertical(bb)) if stm == chess.BLACK else (lambda bb: bb)
        x.ours, x.theirs = o(b.occupied_co[stm]), o(b.occupied_co[not stm])
        x.pawns, x.knights, x.bishops = o(b.pawns), o(b.knights), o(b.bishops)
        x.rooks, x.queens = o(b.rooks), o(b.queens)
        x.our_king, x.their_king = o(b.kings & b.occupied_co[stm]), o(b.kings & b.occupied_co[not stm])
        # lc0 marks en passant (only when an enemy pawn could capture) with a
        # pawn bit on rank 8 of the side-to-move's board.
        x.ep = 0
        if b.ep_square is not None and b.has_pseudo_legal_en_passant():
            x.ep = 1 << (56 + chess.square_file(b.ep_square))
        # (we 000, we 00, they 000, they 00)
        x.cast = (b.has_queenside_castling_rights(stm), b.has_kingside_castling_rights(stm),
                  b.has_queenside_castling_rights(not stm), b.has_kingside_castling_rights(not stm))
        x.flipped = stm == chess.BLACK
        return x

    def mirrored(self):
        y = LBoard()
        y.ours, y.theirs = rev_bytes(self.theirs), rev_bytes(self.ours)
        for k in ("pawns", "knights", "bishops", "rooks", "queens"):
            setattr(y, k, rev_bytes(getattr(self, k)))
        y.our_king, y.their_king = rev_bytes(self.their_king), rev_bytes(self.our_king)
        y.ep = rev_bytes(self.ep)
        y.cast = (self.cast[2], self.cast[3], self.cast[0], self.cast[1])
        y.flipped = not self.flipped
        return y

    def key(self):
        return (self.ours, self.theirs, self.pawns, self.knights, self.bishops, self.rooks,
                self.queens, self.our_king, self.their_king, self.ep, self.cast, self.flipped)

    def kings(self):
        return self.our_king | self.their_king

    def no_castle(self):
        return not any(self.cast)


class LPos:
    __slots__ = ("board", "rule50", "reps")

    def __init__(self, board, rule50):
        self.board, self.rule50, self.reps = board, rule50, 0


STARTPOS_KEY = LBoard.from_pychess(chess.Board()).key()


def build_history(fen, moves):
    b = chess.Board(fen) if fen else chess.Board()
    hist = [LPos(LBoard.from_pychess(b), b.halfmove_clock)]
    for m in moves:
        b.push_uci(m)
        p = LPos(LBoard.from_pychess(b), b.halfmove_clock)
        hist.append(p)
        # PositionHistory::ComputeLastMoveRepetitions
        if p.rule50 >= 4:
            for idx in range(len(hist) - 5, -1, -2):
                q = hist[idx]
                if q.board.key() == p.board.key():
                    p.reps = 1 + q.reps
                    break
                if q.rule50 < 2:
                    break
    return hist, b


def _compare_transposing(v, t):
    if t & FLIP:
        v = rev_bits_in_bytes(v)
    if t & MIRROR:
        v = rev_bytes(v)
    alt = transpose_bits(v)
    return -1 if v < alt else (1 if v > alt else 0)


def choose_transform(bd):
    if not bd.no_castle():
        return 0
    k = bd.our_king
    t = 0
    if k & 0x0F0F0F0F0F0F0F0F:
        t |= FLIP
        k = rev_bits_in_bytes(k)
    if bd.pawns:
        return t
    if k & 0xFFFFFFFF00000000:
        t |= MIRROR
        k = rev_bytes(k)
    if k & 0xE0C08000:
        t |= TRANSPOSE
    elif k & 0x10204080:
        rooks_only = bd.rooks & ~bd.queens
        bishops_only = bd.bishops & ~bd.queens
        for v in (bd.ours | bd.theirs, bd.ours, bd.kings(), bd.queens, rooks_only,
                  bd.knights, bishops_only):
            c = _compare_transposing(v, t)
            if c == -1:
                return t
            if c == 1:
                return t | TRANSPOSE
    return t


def encode_planes(fmt, hist, history_planes=8):
    """Returns (masks[112], vals[112], transform)."""
    masks = [0] * 112
    vals = [1.0] * 112
    aux = 104
    last = hist[-1]
    bd = last.board
    we_black = bd.flipped
    canon = is_canonical(fmt)
    t = choose_transform(bd) if canon else 0
    if fmt == CLASSICAL:
        c = bd.cast
        if c[0]:
            masks[aux + 0] = ALL
        if c[1]:
            masks[aux + 1] = ALL
        if c[2]:
            masks[aux + 2] = ALL
        if c[3]:
            masks[aux + 3] = ALL
    else:
        c = bd.cast  # standard chess: rooks on files a and h
        masks[aux + 0] = ((1 << 0) if c[0] else 0) | ((1 << 56) if c[2] else 0)
        masks[aux + 1] = ((1 << 7) if c[1] else 0) | ((1 << 63) if c[3] else 0)
    if canon:
        masks[aux + 4] = bd.ep
    elif we_black:
        masks[aux + 4] = ALL
    masks[aux + 5] = ALL
    vals[aux + 5] = float(np.float32(last.rule50 / 100.0)) if is_hecto(fmt) else float(last.rule50)
    if is_arma(fmt) and we_black:
        masks[aux + 6] = ALL
    masks[aux + 7] = ALL
    castl = bd.cast
    skip_non_repeats = fmt in (CANON_V2, CANON_V2_ARMA)
    flip = False
    hidx = len(hist) - 1
    i = 0
    guard = 0
    while i < min(history_planes, 8):
        guard += 1
        if guard > 10000:
            raise RuntimeError("lc0 would hang here (trap 5: FEN-only position with a nonzero "
                               "halfmove clock under a V2 canonical format)")
        pos = hist[hidx if hidx >= 0 else 0]
        b = pos.board.mirrored() if flip else pos.board
        if canon and b.cast != castl:
            break
        if canon and hidx != len(hist) - 1 and b.ep:
            break
        if hidx < 0 and pos.board.key() == STARTPOS_KEY:
            break
        reps = pos.reps
        if skip_non_repeats and reps == 0 and i > 0:
            if hidx > 0:
                flip = not flip
            if pos.rule50 == 0:
                break
            hidx -= 1
            continue
        base = i * 13
        pw = b.pawns
        masks[base + 0] = b.ours & pw
        masks[base + 1] = b.ours & b.knights
        masks[base + 2] = b.ours & b.bishops & ~b.queens
        masks[base + 3] = b.ours & b.rooks & ~b.queens
        masks[base + 4] = b.ours & b.queens
        masks[base + 5] = b.our_king
        masks[base + 6] = b.theirs & pw
        masks[base + 7] = b.theirs & b.knights
        masks[base + 8] = b.theirs & b.bishops & ~b.queens
        masks[base + 9] = b.theirs & b.rooks & ~b.queens
        masks[base + 10] = b.theirs & b.queens
        masks[base + 11] = b.their_king
        if reps >= 1:
            masks[base + 12] = ALL
        if hidx < 0 and b.ep:
            idx = (b.ep & -b.ep).bit_length() - 1
            if idx < 8:
                masks[base + 0] = (masks[base + 0] + ((0x0000000000000100 - 0x0000000001000000) << idx)) & M64
            else:
                masks[base + 6] = (masks[base + 6] + ((0x0001000000000000 - 0x0000000100000000) << (idx - 56))) & M64
        if hidx > 0:
            flip = not flip
        if canon and pos.rule50 == 0:
            break
        i += 1
        hidx -= 1
    if t:
        for k in range(aux + 5):
            v = masks[k]
            if v == 0 or v == ALL:
                continue
            if t & FLIP:
                v = rev_bits_in_bytes(v)
            if t & MIRROR:
                v = rev_bytes(v)
            if t & TRANSPOSE:
                v = transpose_bits(v)
            masks[k] = v
    return masks, vals, t


def _sq_transform(sq, t):
    f, r = sq & 7, sq >> 3
    if t & (MIRROR | TRANSPOSE):
        r ^= 7
    if t & (FLIP | TRANSPOSE):
        f ^= 7
    return r * 8 + f


def move_index(board, move, t):
    """lc0 MoveToNNIndex for a python-chess move on `board` (before it)."""
    frm, to = move.from_square, move.to_square
    if board.is_castling(move):  # lc0 encodes castling as king takes own rook
        rank = chess.square_rank(frm)
        to = chess.square(7 if board.is_kingside_castling(move) else 0, rank)
    if board.turn == chess.BLACK:
        frm ^= 56
        to ^= 56
    frm, to = _sq_transform(frm, t), _sq_transform(to, t)
    s = chess.square_name(frm) + chess.square_name(to)
    if move.promotion in (chess.QUEEN, chess.ROOK, chess.BISHOP):
        s += chess.piece_symbol(move.promotion)
    return NN_INDEX[s]


def masks_to_planes(masks, vals):
    out = np.zeros((112, 64), dtype=np.float32)
    for p, m in enumerate(masks):
        if m:
            bits = np.array([(m >> i) & 1 for i in range(64)], dtype=np.float32)
            out[p] = bits * np.float32(vals[p])
    return out.reshape(112, 8, 8)


def encode(fen=None, moves=(), input_format=CANON_V2):
    """-> planes [112,8,8] float32, legal policy indices, legal UCI moves
    (in python-chess's legal-move order)."""
    hist, b = build_history(fen, list(moves))
    masks, vals, t = encode_planes(input_format, hist)
    legal = list(b.legal_moves)
    return masks_to_planes(masks, vals), [move_index(b, m, t) for m in legal], [m.uci() for m in legal]
