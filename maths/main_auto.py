import telebot
from telebot import types
import new  # Your file containing the Draw class
import numpy as np
import re
import ollama
import json

# --- CONFIGURATION ---
API = "8690254124:AAG4hFS89yHbsEcNT3Wsfoa6io1jlVUAGgI"
bot = telebot.TeleBot(token=API)
OLLAMA_MODEL = "qwen2.5:0.5b"

# --- OLLAMA: respond with free text ---
def ollama_respond(text, model=OLLAMA_MODEL):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return response["message"]["content"]

# --- OLLAMA: extract CNC equation as JSON ---
SYSTEM_DRAW_PROMPT = """\
You are a CNC math parser. Convert user input to parametric equations with parameter t.

RULES:
1. If the user asks to draw/plot/graph something, return ONLY valid JSON (no markdown, no explanation):
   {"x_expr": "expression for x(t)", "y_expr": "expression for y(t)", "t_start": 0, "t_end": 6.28, "n_points": 300, "r": 10}
2. Use Python syntax: t**2 (not t^2), math functions: sin(t), cos(t), sqrt(t), exp(t), pi, abs(t)
3. Examples:
   - "y = x^2" -> {"x_expr": "t", "y_expr": "t**2", "t_start": 0, "t_end": 6.28, "n_points": 300, "r": 10}
   - "x = y^2 + 5" -> {"x_expr": "t**2 + 5", "y_expr": "t", "t_start": 0, "t_end": 6.28, "n_points": 300, "r": 10}
   - "circle radius 5" -> {"x_expr": "5*cos(t)", "y_expr": "5*sin(t)", "t_start": 0, "t_end": 6.28, "n_points": 300, "r": 5}
   - "draw a heart" -> {"x_expr": "16*sin(t)**3", "y_expr": "13*cos(t)-5*cos(2*t)-2*cos(3*t)-cos(4*t)", "t_start": 0, "t_end": 6.28, "n_points": 300, "r": 10}
4. If NOT a draw request, reply exactly: NOT_DRAW
5. NO code blocks, NO explanations, ONLY the JSON or NOT_DRAW.
"""

def ollama_parse_draw(text, model=OLLAMA_MODEL):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_DRAW_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response["message"]["content"]

# --- FALLBACK: Regex-based parser for simple equations ---
def parse_simple_equation(text):
    """Fallback parser for basic equations like y = x^2, x = y^2 + 5.
    Returns JSON string or None if parsing fails."""
    import re
    text = text.strip().lower()

    # Match patterns like "y = x^2", "x = y^2 + 5", etc.
    # Pattern: var = expr where expr may contain ^ for power
    pattern = r'^\s*([xy])\s*=\s*(.+?)\s*$'
    match = re.match(pattern, text)
    if not match:
        return None

    dependent_var = match.group(1)  # y or x
    expr = match.group(2).strip()

    # Replace ^ with ** for Python power operator
    expr = re.sub(r'\^', '**', expr)

    # Validate expression contains only safe characters
    if not re.match(r'^[xy\d\+\-\*\/\(\)\.\s\*\*]+$', expr):
        return None

    # Convert to parametric form
    # If y = f(x), then x_expr = "t", y_expr = "t converted"
    # If x = f(y), then x_expr = "t converted", y_expr = "t"
    if dependent_var == 'y':
        x_expr = "t"
        y_expr = expr.replace('x', 't')
    else:  # dependent_var == 'x'
        x_expr = expr.replace('y', 't')
        y_expr = "t"

    result = {
        "x_expr": x_expr,
        "y_expr": y_expr,
        "t_start": 0,
        "t_end": 6.28,
        "n_points": 300,
        "r": 10
    }
    return json.dumps(result)

# --- Add authorized Telegram IDs here (Get yours from @userinfobot) ---
ALLOWED_USERS = [8058658801]

# --- SECURITY CHECK ---
def is_authorized(message):
    print(message.from_user.id)
    return message.from_user.id in ALLOWED_USERS

# --- UTILITY: PARSE ARGUMENTS ---
def get_params(text):
    parts = text.split()
    try:
        if len(parts) < 4:
            return None, "Format: `/shape X Y Radius`"
        vals = [float(i) for i in parts[1:4]]
        return vals, None
    except ValueError:
        return None, "Please use numbers for X, Y, and Radius."

# --- COMMAND: START / MANUAL ---
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    if not is_authorized(message):
        bot.reply_to(message, "Access Denied. Your ID is not whitelisted.")
        return

    manual = (
        "CNC Virtual Controller v3.0 (AI-Powered)\n\n"
        f"Welcome, {message.from_user.first_name}. You can control the CNC via commands, images, or natural language.\n\n"
        "1. Natural Language (AI)\n"
        "Just type what you want: 'draw a heart', 'y equals x squared', etc.\n"
        "Ollama will understand and draw it for you.\n"
        "You can also ask general questions and I'll answer them.\n\n"
        "2. Manual Equation\n"
        "/draw x_expr | y_expr (e.g., /draw r*cos(t) | r*sin(t) | 200 | 0 | 6.28)\n\n"
        "3. Preset Shapes\n"
        "/shape X Y Radius - /circle, /heart_curve, /spiral, /butterfly ...\n\n"
        "4. Image Mode (Auto-Trace)\n"
        "Send a Photo: the bot detects edges and generates CNC toolpaths.\n\n"
        "Ensure images are clear and have high contrast for best results."
    )
    bot.reply_to(message, manual, parse_mode='Markdown')

# --- FREE-FORM EQUATION HANDLER (/draw) ---
@bot.message_handler(commands=['draw'])
def handle_free_draw(message):
    if not is_authorized(message): return

    text = message.text.strip()
    parts_str = text.replace('/draw', '', 1).strip()
    parts = [p.strip() for p in parts_str.split('|')]
    if len(parts) < 2:
        bot.reply_to(message,
            "Usage: /draw x_expr | y_expr | [points=300] | [t_start=0] | [t_end=2*pi]\n"
            "Vars: t (param), r (scale, default 10)\n"
            "Example: /draw r*cos(t) | r*sin(t) | 200 | 0 | 2*pi",
            parse_mode='Markdown')
        return

    x_expr = parts[0]
    y_expr = parts[1]
    n_points = int(parts[2]) if len(parts) > 2 and parts[2] else 300
    t_start = eval(parts[3], {"__builtins__": {}}, {"pi": np.pi, "e": np.e, "np": np}) if len(parts) > 3 and parts[3] else 0
    t_end = eval(parts[4], {"__builtins__": {}}, {"pi": np.pi, "e": np.e, "np": np}) if len(parts) > 4 and parts[4] else 2 * np.pi

    try:
        t = np.linspace(t_start, t_end, n_points)
        r = 10
        safe_dict = {"t": t, "r": r, "np": np, "pi": np.pi, "e": np.e}
        safe_dict.update({name: getattr(np, name) for name in ['sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 'log10', 'abs', 'arcsin', 'arccos', 'arctan', 'pi', 'e']})

        xc = eval(x_expr, {"__builtins__": {}}, safe_dict)
        yc = eval(y_expr, {"__builtins__": {}}, safe_dict)

        if isinstance(xc, (int, float)):
            xc = np.full_like(t, xc)
        if isinstance(yc, (int, float)):
            yc = np.full_like(t, yc)

        new.solution(xc, yc, t, f"Custom: {x_expr}, {y_expr}")
        bot.send_message(message.chat.id, f"Drew: x={x_expr}, y={y_expr}")
    except Exception as e:
        bot.reply_to(message, f"Error evaluating equation: {e}")

# --- DYNAMIC SHAPE HANDLER (preset shapes) ---
ALL_SHAPES = [
    'circle', 'heart_curve', 'petal_rose', 'lissajous', 'butterfly',
    'spiral', 'cardioid', 'astroid', 'epitrochoid', 'hypotrochoid',
    'rhodonea', 'limacon', 'cycloid', 'deltoid', 'logarithmic_spiral'
]

@bot.message_handler(commands=ALL_SHAPES)
def handle_all_drawings(message):
    if not is_authorized(message): return

    cmd = message.text.split()[0].replace('/', '').lower()
    params, error = get_params(message.text)

    if error:
        bot.reply_to(message, f"{error}", parse_mode='Markdown')
        return

    x, y, r = params
    bot.send_message(message.chat.id, f"Simulating {cmd}...")

    try:
        worker = new.Draw(x, y, r)
        drawing_func = getattr(worker, cmd)
        drawing_func()
        bot.send_message(message.chat.id, f"{cmd} complete.")
    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")

# --- IMAGE HANDLER (CARTOON TRACE) ---
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    if not is_authorized(message): return

    bot.reply_to(message, "Image received! Converting to CNC paths...")

    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        image_name = "input_image.jpg"
        with open(image_name, 'wb') as f:
            f.write(downloaded_file)

        worker = new.Draw(0, 0, 15)
        worker.draw_cartoon(image_name)

        bot.send_message(message.chat.id, "Image trace simulation complete!")
    except Exception as e:
        bot.reply_to(message, f"Error processing image: {e}")

# --- NLP HANDLER: forward all text to Ollama ---
@bot.message_handler(func=lambda message: True)
def handle_text(message):
    if not is_authorized(message): return

    user_text = message.text.strip()
    if user_text.startswith('/'):
        bot.reply_to(message, "Unknown command. Use /help for instructions.")
        return

    bot.send_chat_action(message.chat.id, 'typing')

    # Try Ollama first
    raw = ollama_parse_draw(user_text)
    data = None

    # Try to parse Ollama response as JSON
    if "NOT_DRAW" not in raw:
        try:
            cleaned = raw.strip()
            # Strip markdown code fences
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            data = json.loads(cleaned)
        except (json.JSONDecodeError, Exception):
            data = None

    # Fallback: try regex parser if Ollama failed
    if data is None:
        fallback_json = parse_simple_equation(user_text)
        if fallback_json:
            try:
                data = json.loads(fallback_json)
            except (json.JSONDecodeError, Exception):
                data = None

    # If we have valid draw data, render it
    if data and "x_expr" in data and "y_expr" in data:
        try:
            t = np.linspace(data.get("t_start", 0), data.get("t_end", 2*np.pi), data.get("n_points", 300))
            r = data.get("r", 10)
            safe_dict = {"t": t, "r": r, "np": np, "pi": np.pi, "e": np.e}
            safe_dict.update({name: getattr(np, name) for name in ['sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 'log10', 'abs', 'arcsin', 'arccos', 'arctan']})
            xc = eval(data["x_expr"], {"__builtins__": {}}, safe_dict)
            yc = eval(data["y_expr"], {"__builtins__": {}}, safe_dict)
            if isinstance(xc, (int, float)):
                xc = np.full_like(t, xc)
            if isinstance(yc, (int, float)):
                yc = np.full_like(t, yc)
            new.solution(xc, yc, t, f"Custom: {data['x_expr']}, {data['y_expr']}")
            bot.send_message(message.chat.id, f"Drew: x={data['x_expr']}, y={data['y_expr']}")
            return
        except Exception:
            pass

    # Not a draw -> answer with Ollama
    answer = ollama_respond(user_text)
    bot.reply_to(message, answer)

bot.polling()
