
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_pymongo import PyMongo
from datetime import datetime, timedelta
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import chess
import chess.pgn
from io import StringIO
from bson.objectid import ObjectId
import jwt
import json
import time
from typing import List, Tuple, Optional, Dict
from functools import wraps

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/chessAnalysis"
app.config["SECRET_KEY"] = "{~DKvCX5dJ/j.r!k3~'DF?mY59k75]pN"  # Move to .env in production
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}}, supports_credentials=True, send_wildcard=True)
mongo = PyMongo(app)

# Custom error handler to ensure CORS headers
@app.errorhandler(Exception)
def handle_error(error):
    response = make_response(jsonify({"error": str(error)}), 500)
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ChessAnalyzer class
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

PST_PAWN = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 15, 25, 25, 15,  5,  5,
    0,  0, 10, 20, 20, 10,  0,  0,
    5, -5,  0,  0,  0,  0, -5,  5,
    5, 10, 10,-10,-10, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30, 10, 15, 20, 20, 15, 10,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 15, 15, 10,  0,-10,
    -10,  0, 10, 15, 15, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

PST_ROOK = [
    0,  0,  0,  5,  5,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0, 10, 10,  0,  0,  0
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  5,  0,-10,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -5,  0,  5, 10, 10,  5,  0, -5,
    -5,  0,  5, 10, 10,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

PST_KING_MID = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

PST_KING_END = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

PST = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING_MID
}

class TranspositionTable:
    def __init__(self):
        self.table: Dict[str, float] = {}

    def store(self, fen: str, eval_score: float) -> None:
        self.table[fen] = eval_score

    def lookup(self, fen: str) -> Optional[float]:
        return self.table.get(fen)

    def size(self) -> int:
        return len(self.table)

class ChessAnalyzer:
    def __init__(self, eval_file: str = "cleaned_combined_data.json"):
        self.board = chess.Board()
        self.eval_file = eval_file
        self.transposition_table = TranspositionTable()
        self.stockfish_evals = self.load_stockfish_evals()
        self.history = {}
        self.killer_moves = {}
        self.move_history_heuristic = {}

    def validate_fen(self, fen: str) -> bool:
        try:
            board = chess.Board(fen)
            white_kings = sum(1 for square in chess.SQUARES if board.piece_at(square) == chess.Piece(chess.KING, chess.WHITE))
            black_kings = sum(1 for square in chess.SQUARES if board.piece_at(square) == chess.Piece(chess.KING, chess.BLACK))
            return white_kings == 1 and black_kings == 1 and board.is_valid()
        except ValueError as e:
            print(f"FEN validation error: {fen}, {str(e)}")
            return False

    def load_stockfish_evals(self) -> Dict[str, float]:
        evals = {}
        try:
            with open(self.eval_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    fen = entry.get('fen')
                    cp = entry.get('cp', entry.get('eval', 0))
                    if not isinstance(cp, (int, float)):
                        print(f"Skipping entry with invalid evaluation: {entry}")
                        continue
                    if fen and self.validate_fen(fen):
                        eval_score = cp / 100.0
                        evals[fen] = eval_score
                        self.transposition_table.store(fen, eval_score)
                    else:
                        print(f"Skipping invalid FEN: {fen}")
                print(f"Loaded {len(evals)} valid positions into transposition table.")
        except FileNotFoundError:
            print(f"Warning: {self.eval_file} not found. Using default evaluations.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.eval_file}. Using default evaluations.")
        except Exception as e:
            print(f"Unexpected error loading {self.eval_file}: {e}")
        return evals

    def evaluate_position(self) -> float:
        fen = self.board.fen()
        if not self.validate_fen(fen):
            print(f"Error: Invalid FEN: {fen}")
            return 0.0

        if self.board.is_checkmate():
            return -float('inf') if self.board.turn == chess.WHITE else float('inf')
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0

        cached_eval = self.transposition_table.lookup(fen)
        if cached_eval is not None:
            return cached_eval

        score = 0.0
        total_material = 0
        mobility_score = 0
        center_control = 0
        pawn_structure = 0
        check_repetition = 0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                pst_table = PST[piece.piece_type]
                pst_index = square if piece.color == chess.WHITE else chess.square_mirror(square)
                pst_value = pst_table[pst_index]
                multiplier = 1 if piece.color == chess.WHITE else -1
                score += multiplier * (value + pst_value)
                if piece.piece_type != chess.KING:
                    total_material += value

                if piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    legal_moves = len([m for m in self.board.generate_legal_moves(from_mask=chess.BB_SQUARES[square])])
                    mobility_score += multiplier * legal_moves * 10

                if square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                    if piece.piece_type == chess.PAWN:
                        center_control += multiplier * 20
                    elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        center_control += multiplier * 15

        phase = min(1.0, total_material / 4000)
        if phase < 1.0:
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.KING:
                    pst_index = square if piece.color == chess.WHITE else chess.square_mirror(square)
                    score += (1 if piece.color == chess.WHITE else -1) * PST_KING_END[pst_index] * (1 - phase)

        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            pawns = [s for s in chess.SQUARES if self.board.piece_at(s) and self.board.piece_at(s).piece_type == chess.PAWN and self.board.piece_at(s).color == color]
            pawn_files = [chess.square_file(s) for s in pawns]
            for file in range(8):
                file_pawns = pawn_files.count(file)
                if file_pawns > 1:
                    pawn_structure -= multiplier * 50 * (file_pawns - 1)
                if file_pawns == 0:
                    neighbors = [pawn_files.count(f) for f in [file-1, file+1] if 0 <= f <= 7]
                    if not any(neighbors):
                        pawn_structure -= multiplier * 30

            for pawn in pawns:
                rank = chess.square_rank(pawn)
                file = chess.square_file(pawn)
                is_passed = True
                for f in [file-1, file, file+1]:
                    if 0 <= f <= 7:
                        for r in range(rank + (1 if color == chess.WHITE else -1), (8 if color == chess.WHITE else -1), (1 if color == chess.WHITE else -1)):
                            if self.board.piece_at(chess.square(f, r)) and self.board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and self.board.piece_at(chess.square(f, r)).color != color:
                                is_passed = False
                                break
                if is_passed:
                    pawn_structure += multiplier * (20 + 10 * (rank if color == chess.WHITE else 7 - rank))

        for color in [chess.WHITE, chess.BLACK]:
            king_square = self.board.king(color)
            if king_square is None:
                continue
            multiplier = 1 if color == chess.WHITE else -1
            attacks = self.board.attackers(not color, king_square)
            if attacks:
                score -= multiplier * 80 * len(attacks)
            rank = chess.square_rank(king_square)
            file = chess.square_file(king_square)
            shield_squares = []
            if color == chess.WHITE and rank < 7:
                shield_squares = [chess.square(file + i, rank + 1) for i in [-1, 0, 1] if 0 <= file + i <= 7]
            elif color == chess.BLACK and rank > 0:
                shield_squares = [chess.square(file + i, rank - 1) for i in [-1, 0, 1] if 0 <= file + i <= 7]
            for square in shield_squares:
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    score += multiplier * 40

        if self.board.is_check():
            move_key = (fen, str(self.board.peek()))
            check_repetition = self.history.get(move_key, 0)
            score -= check_repetition * 150

        score += mobility_score + center_control + pawn_structure
        eval_score = score / 100.0
        self.transposition_table.store(fen, eval_score)
        return eval_score

    def is_square_safe(self, square: chess.Square, color: chess.Color) -> bool:
        return not bool(self.board.attackers(not color, square))

    def get_capture_value(self, move: chess.Move) -> Tuple[int, int]:
        capturer = self.board.piece_at(move.from_square)
        capturer_val = PIECE_VALUES.get(capturer.piece_type, 0) if capturer else 0
        if move in self.board.legal_moves and self.board.is_capture(move):
            captured = self.board.piece_at(move.to_square)
            captured_val = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
            return (captured_val, capturer_val)
        return (0, capturer_val)

    def can_castle(self, move: chess.Move) -> bool:
        if not self.board.is_castling(move):
            return False
        king_square = self.board.king(self.board.turn)
        if king_square is None:
            return False
        rook_square = move.to_square
        path_squares = chess.SquareSet.between(king_square, rook_square) | {king_square, rook_square}
        for square in path_squares:
            if self.board.attackers(not self.board.turn, square):
                return False
        self.board.push(move)
        safe = self.is_square_safe(self.board.king(self.board.turn), self.board.turn)
        self.board.pop()
        return safe

    def is_king_move_safe(self, move: chess.Move) -> bool:
        if self.board.piece_at(move.from_square).piece_type != chess.KING:
            return True
        if self.board.is_check():
            self.board.push(move)
            safe = not self.board.is_check()
            self.board.pop()
            return safe
        return self.is_square_safe(move.to_square, self.board.turn)

    def get_best_move(self, depth: int = 3, time_limit: float = 2.0) -> Tuple[Optional[chess.Move], float]:
        def quiescence(board: chess.Board, alpha: float, beta: float, depth_limit: int = 6) -> float:
            if depth_limit <= 0:
                return self.evaluate_position()
            stand_pat = self.evaluate_position()
            if stand_pat == 0.0 and board.king(board.turn) is None:
                return stand_pat
            if board.turn == chess.WHITE:
                if stand_pat >= beta:
                    return beta
                alpha = max(alpha, stand_pat)
                moves = []
                for move in board.legal_moves:
                    if board.is_capture(move) or move.promotion:
                        captured_val, capturer_val = self.get_capture_value(move)
                        score = captured_val - capturer_val / 100.0
                        if move.promotion == chess.QUEEN:
                            score += 900
                        moves.append((move, score))
                moves.sort(key=lambda x: x[1], reverse=True)
                for move, _ in moves:
                    board.push(move)
                    score = quiescence(board, alpha, beta, depth_limit - 1)
                    board.pop()
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        return beta
                return alpha
            else:
                if stand_pat <= alpha:
                    return alpha
                beta = min(beta, stand_pat)
                moves = []
                for move in board.legal_moves:
                    if board.is_capture(move) or move.promotion:
                        captured_val, capturer_val = self.get_capture_value(move)
                        score = captured_val - capturer_val / 100.0
                        if move.promotion == chess.QUEEN:
                            score += 900
                        moves.append((move, score))
                moves.sort(key=lambda x: x[1], reverse=True)
                for move, _ in moves:
                    board.push(move)
                    score = quiescence(board, alpha, beta, depth_limit - 1)
                    board.pop()
                    beta = min(beta, score)
                    if beta <= alpha:
                        return alpha
                return beta

        def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, start_time: float) -> Tuple[float, Optional[chess.Move]]:
            if time.time() - start_time > time_limit:
                return self.evaluate_position(), None
            if depth == 0 or board.is_game_over():
                return quiescence(board, alpha, beta), None

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return self.evaluate_position(), None

            move_scores = []
            killers = self.killer_moves.get(depth, [])
            for move in legal_moves:
                score = 0
                if move in killers:
                    score += 2000
                if board.is_capture(move):
                    captured_val, capturer_val = self.get_capture_value(move)
                    score += 1500 + captured_val - capturer_val / 100.0
                if move.promotion == chess.QUEEN:
                    score += 1000
                if board.is_castling(move) and self.can_castle(move):
                    score += 900
                if board.gives_check(move):
                    score += 600
                piece = board.piece_at(move.from_square)
                if piece:
                    score += PST[piece.piece_type][move.to_square]
                move_key = (board.fen(), str(move))
                score += self.move_history_heuristic.get(move_key, 0)
                score -= self.history.get(move_key, 0) * 100
                move_scores.append((move, score))

            move_scores.sort(key=lambda x: x[1], reverse=True)

            best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
            best_move = None

            for move, _ in move_scores[:20]:
                if not self.is_king_move_safe(move):
                    continue
                board.push(move)
                score, _ = alpha_beta(board, depth - 1, alpha, beta, start_time)
                board.pop()
                move_key = (board.fen(), str(move))
                self.history[move_key] = self.history.get(move_key, 0) + 1
                if board.turn == chess.WHITE:
                    if score > best_score:
                        best_score = score
                        best_move = move
                        self.move_history_heuristic[move_key] = self.move_history_heuristic.get(move_key, 0) + depth * depth
                        if len(killers) < 2:
                            killers.append(move)
                        else:
                            killers[1] = killers[0]
                            killers[0] = move
                        self.killer_moves[depth] = killers
                    alpha = max(alpha, best_score)
                else:
                    if score < best_score:
                        best_score = score
                        best_move = move
                        self.move_history_heuristic[move_key] = self.move_history_heuristic.get(move_key, 0) + depth * depth
                        if len(killers) < 2:
                            killers.append(move)
                        else:
                            killers[1] = killers[0]
                            killers[0] = move
                        self.killer_moves[depth] = killers
                    beta = min(beta, score)
                if beta <= alpha:
                    self.move_history_heuristic[move_key] = self.move_history_heuristic.get(move_key, 0) + depth * depth
                    break

            return best_score, best_move

        legal_moves = list(self.board.legal_moves)
        dynamic_depth = depth + 1 if len(legal_moves) < 10 else depth
        start_time = time.time()
        score, move = alpha_beta(self.board, dynamic_depth, -float('inf'), float('inf'), start_time)
        return move, score

    def classify_move(self, move: chess.Move, best_move: chess.Move, best_score: float, played_score: float) -> Tuple[str, str]:
        eval_diff = min(abs(best_score - played_score), 10.0)
        if move == best_move or eval_diff < 0.1:
            return "Best", "This is the optimal move."
        elif eval_diff < 0.4:
            return "Good", "A solid move, close to the best."
        elif eval_diff < 1.5:
            return "Mistake", f"Suboptimal move, losing {eval_diff:.2f} pawns."
        else:
            return "Blunder", f"Significant error, losing {eval_diff:.2f} pawns."

    def analyze_game(self, game: chess.pgn.Game) -> List[Dict]:
        analysis = []
        self.board.reset()
        move_number = 1

        for move in game.mainline_moves():
            player = "White" if self.board.turn == chess.WHITE else "Black"
            move_san = self.board.san(move)
            fen_before = self.board.fen()

            best_move, best_score = self.get_best_move(depth=3, time_limit=2.0)
            self.board.push(move)
            played_score = self.evaluate_position()
            self.board.pop()
            classification, explanation = self.classify_move(move, best_move, best_score, played_score)

            analysis.append({
                "move_number": move_number // 2 + 1,
                "player": player,
                "move_played": move_san,
                "best_move": self.board.san(best_move) if best_move else "None",
                "board_fen": fen_before,
                "evaluation": played_score,
                "predicted_best_move": self.board.san(best_move) if best_move else "None",
                "predicted_evaluation": best_score,
                "classification": classification,
                "comment": explanation
            })
            self.board.push(move)
            move_number += 1

        return analysis

    def predict_next_move(self, fen: str) -> Tuple[str, float]:
        if not self.validate_fen(fen):
            print(f"Error: Invalid FEN: {fen}")
            return "None", 0.0
        try:
            self.board.set_fen(fen)
        except ValueError as e:
            print(f"Error: Invalid FEN: {fen}, {str(e)}")
            return "None", 0.0
        move, score = self.get_best_move(depth=3, time_limit=2.0)
        return self.board.san(move) if move else "None", score

# Initialize ChessAnalyzer
analyzer = ChessAnalyzer(eval_file="b:\\chess_engine_4\\cleaned_combined_data.json")

# JWT token required decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            response = jsonify({'error': 'Token is missing'})
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            return response, 401
        try:
            token = token.split()[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = mongo.db.users.find_one({'username': data['username']})
            if not current_user:
                response = jsonify({'error': 'User not found'})
                response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
                return response, 401
        except Exception as e:
            response = jsonify({'error': f'Invalid token: {str(e)}'})
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            return response, 401
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/api/chesscom/games', methods=['GET'])
def chesscom_games():
    username = request.args.get('username')
    if not username:
        response = jsonify({"error": "No Chess.com username provided"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400
    
    username = username.lower()
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept": "application/ld+json"
    }

    try:
        archives_response = requests.get(archives_url, headers=headers)
        if archives_response.status_code == 429:
            response = jsonify({"error": "Chess.com API rate limit exceeded. Try again later."})
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            return response, 429
        if archives_response.status_code != 200:
            response = jsonify({"error": f"Error fetching archives from Chess.com: {archives_response.status_code}"})
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            return response, archives_response.status_code
        
        archives_data = archives_response.json()
        archives_list = archives_data.get("archives", [])
        if not archives_list:
            response = jsonify({"error": "No games found for this username"})
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            return response, 404
        
        pgn_list = []
        for archive_url in archives_list[-3:]:
            games_response = requests.get(archive_url, headers=headers)
            if games_response.status_code == 429:
                response = jsonify({"error": "Chess.com API rate limit exceeded. Try again later."})
                response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
                return response, 429
            if games_response.status_code == 200:
                games_data = games_response.json()
                games = games_data.get("games", [])
                pgn_list.extend([game.get("pgn", "") for game in games if game.get("pgn")])
        
        pgn_list = pgn_list[-10:] if len(pgn_list) > 10 else pgn_list
        print('Returning games:', pgn_list[:2])
        response = jsonify({"games": pgn_list})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
    except requests.RequestException as e:
        print(f"Error fetching Chess.com games: {str(e)}")
        response = jsonify({"error": "Failed to connect to Chess.com API"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 500

@app.route('/api/save-analysis', methods=['POST'])
@token_required
def save_analysis(current_user):
    data = request.get_json()
    print('Received request data:', data)
    username = data.get("username")
    pgn = data.get("pgn")
    analysis = data.get("analysis", [])
    last_viewed_move = data.get("last_viewed_move", 0)
    comments = data.get("comments", [])

    if not username or not pgn:
        error_msg = "Missing username or PGN"
        print('Validation failed:', error_msg)
        response = jsonify({"error": error_msg})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400

    if username != current_user['username']:
        response = jsonify({'error': 'Unauthorized: Username mismatch'})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 403

    analysis_entry = {
        "username": username,
        "pgn": pgn,
        "analysis": analysis,
        "last_viewed_move": last_viewed_move,
        "comments": comments,
        "timestamp": datetime.now().isoformat()
    }
    try:
        result = mongo.db.analysis_history.insert_one(analysis_entry)
        print('Saved analysis with ID:', str(result.inserted_id))
        response = jsonify({"message": "Analysis saved", "id": str(result.inserted_id)})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
    except Exception as e:
        print(f"Error saving analysis: {str(e)}")
        response = jsonify({"error": "Error saving analysis"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 500

@app.route('/api/analysis-history/<username>', methods=['GET'])
@token_required
def get_analysis_history(current_user, username):
    if not username:
        response = jsonify({'error': 'Username is required'})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400
    if username != current_user['username']:
        response = jsonify({'error': 'Unauthorized: Username mismatch'})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 403

    try:
        history = mongo.db.analysis_history.find({"username": username}).sort("timestamp", -1).limit(10)
        history_list = [
            {
                "id": str(entry["_id"]),
                "pgn": entry["pgn"],
                "analysis": entry["analysis"],
                "last_viewed_move": entry["last_viewed_move"],
                "comments": entry.get("comments", []),
                "timestamp": entry["timestamp"]
            }
            for entry in history
        ]
        print('Returning analysis history for', username, ':', history_list)
        response = jsonify({"history": history_list})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
    except Exception as e:
        print(f"Error fetching analysis history: {str(e)}")
        response = jsonify({"error": "Error fetching analysis history"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 500

@app.route('/api/analyze_game', methods=['POST'])
@token_required
def analyze_game(current_user):
    data = request.get_json()
    username = data.get("username")
    pgn_string = data.get("pgn")

    if not username or not pgn_string:
        response = jsonify({"error": "Missing username or PGN"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400

    if username != current_user['username']:
        response = jsonify({'error': 'Unauthorized: Username mismatch'})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 403

    pgn_io = StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    if not game:
        response = jsonify({"error": "Invalid PGN"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400

    try:
        analysis = analyzer.analyze_game(game)
        print('Game analysis (first two moves):', analysis[:2])
        response = jsonify({"analysis": analysis})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
    except Exception as e:
        print(f"Error analyzing game: {str(e)}")
        response = jsonify({"error": "Error analyzing game"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 500

@app.route('/api/update-last-viewed/<analysis_id>', methods=['POST'])
@token_required
def update_last_viewed(current_user, analysis_id):
    data = request.get_json()
    last_viewed_move = data.get("last_viewed_move")
    comments = data.get("comments")
    
    if last_viewed_move is None:
        response = jsonify({"error": "Missing last_viewed_move"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400

    try:
        analysis = mongo.db.analysis_history.find_one({"_id": ObjectId(analysis_id)})
        if not analysis:
            response = jsonify({"error": "Analysis not found"})
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            return response, 404
        if analysis["username"] != current_user["username"]:
            response = jsonify({"error": "Unauthorized: Username mismatch"})
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            return response, 403

        update_data = {"last_viewed_move": last_viewed_move}
        if comments is not None:
            update_data["comments"] = comments
        
        result = mongo.db.analysis_history.update_one(
            {"_id": ObjectId(analysis_id)},
            {"$set": update_data}
        )
        if result.modified_count == 0:
            response = jsonify({"error": "No changes made"})
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            return response, 400
        print('Updated analysis ID:', analysis_id, 'with', update_data)
        response = jsonify({"message": "Last viewed move updated"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
    except Exception as e:
        print(f"Error updating last viewed move: {str(e)}")
        response = jsonify({"error": "Error updating last viewed move"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    print('Login request data:', data)
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        response = jsonify({"error": "Missing username or password"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400
    
    user = mongo.db.users.find_one({"username": username})
    if not user:
        print(f"Login failed: User {username} not found")
        response = jsonify({"error": "User not found"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 401
    
    if not check_password_hash(user["password"], password):
        print(f"Login failed: Incorrect password for {username}")
        response = jsonify({"error": "Incorrect password"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 401

    try:
        token = jwt.encode({
            'username': user['username'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        response = jsonify({
            "message": "Logged in successfully",
            "token": token,
            "user": {"username": user["username"], "email": user.get("email", "")}
        })
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
    except Exception as e:
        print(f"Error generating JWT token: {str(e)}")
        response = jsonify({"error": "Error generating token"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 500

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        response = jsonify({"error": "Missing fields"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400
    if mongo.db.users.find_one({"username": username}):
        response = jsonify({"error": "Username already exists"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400
    if mongo.db.users.find_one({"email": email}):
        response = jsonify({"error": "Email already exists"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 400
    
    password_hash = generate_password_hash(password)
    user_data = {"username": username, "email": email, "password": password_hash}
    try:
        mongo.db.users.insert_one(user_data)
        print(f"Registered user: {username}")
        response = jsonify({"message": "User registered successfully"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response
    except Exception as e:
        print(f"Error registering user: {str(e)}")
        response = jsonify({"error": "Error registering user"})
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)