import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from datetime import datetime
import tempfile
import io
import re
import json
import math
import traceback

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="AI Simulation Animator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Cute presentation CSS
# =========================================================
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 850;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #7c3aed, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .cute-card {
        border: 1px solid #e5e7eb;
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        border-radius: 18px;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
    }
    .ok-box {
        border: 1px solid #bbf7d0;
        background: #f0fdf4;
        color: #166534;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .warn-box {
        border: 1px solid #fed7aa;
        background: #fff7ed;
        color: #9a3412;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .danger-box {
        border: 1px solid #fecaca;
        background: #fef2f2;
        color: #991b1b;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 800;
    }
</style>
""",
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🧪 AI Simulation Animator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">주제 입력 → GPT 코드 생성 → 파라미터 조절 → Preview → GIF/PNG 다운로드</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="cute-card">
<b>핵심 구조</b><br>
이 웹앱은 주제를 미리 정하지 않습니다. GPT가 주제에 맞게 
<code>PARAM_CONFIG</code>, <code>init_state</code>, <code>update_state</code>, 
<code>draw_frame</code>을 만들어오면, 웹앱은 그 코드를 실행해서 조절 가능한 애니메이션으로 변환합니다.
</div>
""",
    unsafe_allow_html=True
)

# =========================================================
# Utilities
# =========================================================
def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(name):
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name if name else "simulation"


def json_safe(obj):
    """Convert numpy objects and other common values to JSON-safe forms."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def extract_python_code(text):
    """
    If GPT answer includes markdown code blocks, choose the block containing
    PARAM_CONFIG and required functions. Otherwise return original text.
    """
    text = text or ""
    blocks = re.findall(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not blocks:
        return text.strip()

    def score(block):
        s = 0
        for token in [
            "PARAM_CONFIG",
            "def init_state",
            "def update_state",
            "def draw_frame",
            "FIGURE_CONFIG"
        ]:
            if token in block:
                s += 1
        return s

    blocks = sorted(blocks, key=score, reverse=True)
    return blocks[0].strip()


def get_default_code():
    return """PARAM_CONFIG = {
    "rotation_speed": {
        "type": "slider",
        "min": -0.20,
        "max": 0.20,
        "default": 0.045,
        "step": 0.005,
        "unit": "rad/frame",
        "description": "Rotation speed of the educational orbital-like cloud."
    },
    "cloud_size": {
        "type": "slider",
        "min": 0.4,
        "max": 1.8,
        "default": 1.0,
        "step": 0.05,
        "description": "Overall size of the cloud."
    },
    "point_count": {
        "type": "int",
        "min": 200,
        "max": 2000,
        "default": 900,
        "step": 100,
        "description": "Number of points used for the visualization."
    },
    "cloud_color": {
        "type": "color",
        "default": "#7c3aed",
        "description": "Color of the moving cloud."
    }
}

FIGURE_CONFIG = {
    "projection": "3d",
    "figsize": (7.5, 6.5),
    "dpi": 120
}

def init_state(params):
    n = int(params["point_count"])
    rng = np.random.default_rng(3)

    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.normal(0, 0.28, n)
    r = 0.35 + 0.65 * rng.random(n)

    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)

    return {
        "x0": x0,
        "y0": y0,
        "z0": z,
        "x": x0.copy(),
        "y": y0.copy(),
        "z": z.copy()
    }

def update_state(state, params, frame):
    speed = params["rotation_speed"]
    size = params["cloud_size"]

    angle = speed * frame
    c = np.cos(angle)
    s = np.sin(angle)

    x0 = state["x0"] * size
    y0 = state["y0"] * size
    z0 = state["z0"] * size

    state["x"] = c * x0 - s * y0
    state["y"] = s * x0 + c * y0
    state["z"] = z0 + 0.15 * np.sin(angle + 4 * x0)

    return state

def draw_frame(ax, state, params, frame):
    ax.clear()

    ax.scatter(
        state["x"],
        state["y"],
        state["z"],
        s=12,
        alpha=0.45,
        color=params["cloud_color"],
        label="Rotating educational cloud"
    )

    ax.quiver(
        0, 0, 0,
        0, 0, 1.1,
        color="#f59e0b",
        linewidth=3,
        arrow_length_ratio=0.15,
        label="Reference spin axis"
    )

    ax.set_title("Generic 3D Parameter-Controlled Simulation", fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    limit = 1.8 * params["cloud_size"]
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=24, azim=35 + frame * 0.6)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)
"""


def execute_user_code(code):
    """
    Execute GPT-generated code and return namespace.
    This app is meant for trusted educational/demo use.
    """
    namespace = {}
    safe_globals = {
        "np": np,
        "math": math,
        "__builtins__": __builtins__,
    }
    exec(code, safe_globals, namespace)
    return namespace


def validate_namespace(namespace):
    errors = []

    if "PARAM_CONFIG" not in namespace:
        errors.append("PARAM_CONFIG가 없습니다.")
    elif not isinstance(namespace["PARAM_CONFIG"], dict):
        errors.append("PARAM_CONFIG는 dictionary여야 합니다.")

    for func_name in ["init_state", "update_state", "draw_frame"]:
        if func_name not in namespace:
            errors.append(f"{func_name} 함수가 없습니다.")
        elif not callable(namespace[func_name]):
            errors.append(f"{func_name}가 callable 함수가 아닙니다.")

    return errors


def get_figure_config(namespace):
    cfg = namespace.get("FIGURE_CONFIG", {})
    if not isinstance(cfg, dict):
        cfg = {}

    projection = str(cfg.get("projection", "2d")).lower()
    if projection not in ["2d", "3d"]:
        projection = "2d"

    figsize = cfg.get("figsize", (8.0, 5.6))
    dpi = int(cfg.get("dpi", 110))

    try:
        figsize = tuple(figsize)
        if len(figsize) != 2:
            figsize = (8.0, 5.6)
    except Exception:
        figsize = (8.0, 5.6)

    return {
        "projection": projection,
        "figsize": figsize,
        "dpi": dpi
    }


def create_param_widget(name, cfg, key_prefix="param"):
    if not isinstance(cfg, dict):
        cfg = {"type": "slider", "default": 1.0}

    ptype = str(cfg.get("type", "slider")).lower()
    desc = str(cfg.get("description", ""))
    unit = cfg.get("unit", "")
    label = f"{name}"
    if unit:
        label += f" ({unit})"

    key = f"{key_prefix}_{name}"

    if ptype == "checkbox":
        return st.sidebar.checkbox(
            label,
            value=bool(cfg.get("default", False)),
            help=desc,
            key=key
        )

    if ptype == "color":
        return st.sidebar.color_picker(
            label,
            value=str(cfg.get("default", "#7c3aed")),
            help=desc,
            key=key
        )

    if ptype == "text":
        return st.sidebar.text_input(
            label,
            value=str(cfg.get("default", "")),
            help=desc,
            key=key
        )

    if ptype == "select":
        options = cfg.get("options", [])
        if not isinstance(options, (list, tuple)) or len(options) == 0:
            options = ["option_1"]
        default = cfg.get("default", options[0])
        index = list(options).index(default) if default in options else 0
        return st.sidebar.selectbox(
            label,
            options=list(options),
            index=index,
            help=desc,
            key=key
        )

    if ptype == "int":
        default = int(cfg.get("default", 1))
        min_value = int(cfg.get("min", 0))
        max_value = int(cfg.get("max", max(default, 100)))
        step = int(cfg.get("step", 1))
        default = min(max(default, min_value), max_value)
        return st.sidebar.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default,
            step=step,
            help=desc,
            key=key
        )

    if ptype == "number":
        default = float(cfg.get("default", 1.0))
        step = float(cfg.get("step", 0.01))
        return st.sidebar.number_input(
            label,
            value=default,
            step=step,
            format="%.6f",
            help=desc,
            key=key
        )

    if ptype == "log_slider":
        min_value = float(cfg.get("min", 1e-3))
        max_value = float(cfg.get("max", 1e3))
        default = float(cfg.get("default", 1.0))

        min_value = max(min_value, 1e-12)
        max_value = max(max_value, min_value * 10)
        default = min(max(default, min_value), max_value)

        log_min = float(np.log10(min_value))
        log_max = float(np.log10(max_value))
        log_default = float(np.log10(default))

        log_value = st.sidebar.slider(
            label + " log10",
            min_value=log_min,
            max_value=log_max,
            value=log_default,
            step=float(cfg.get("step", 0.05)),
            help=desc,
            key=key
        )

        return float(10 ** log_value)

    # Default: slider
    default = float(cfg.get("default", 1.0))
    min_value = float(cfg.get("min", min(default - 1.0, 0.0)))
    max_value = float(cfg.get("max", max(default + 1.0, 1.0)))
    step = float(cfg.get("step", 0.01))

    if min_value >= max_value:
        min_value, max_value = 0.0, max(default * 2, 1.0)

    default = min(max(default, min_value), max_value)

    return st.sidebar.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default,
        step=step,
        help=desc,
        key=key
    )


def call_init_state(init_state, params):
    return init_state(params)


def call_update_state(update_state, state, params, frame):
    try:
        return update_state(state, params, frame)
    except TypeError:
        return update_state(state, params)


def call_draw_frame(draw_frame, ax, state, params, frame):
    try:
        result = draw_frame(ax, state, params, frame)
    except TypeError:
        result = draw_frame(ax, state, params)
    return result


def make_figure(fig_cfg):
    fig = plt.figure(figsize=fig_cfg["figsize"], dpi=fig_cfg["dpi"])
    if fig_cfg["projection"] == "3d":
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    return fig, ax


def run_state_to_frame(namespace, params, target_frame):
    state = call_init_state(namespace["init_state"], params)

    for frame in range(int(target_frame) + 1):
        state = call_update_state(namespace["update_state"], state, params, frame)

    return state


def render_single_frame(namespace, params, frame):
    fig_cfg = get_figure_config(namespace)
    state = run_state_to_frame(namespace, params, frame)
    fig, ax = make_figure(fig_cfg)
    call_draw_frame(namespace["draw_frame"], ax, state, params, frame)
    fig.tight_layout()
    return fig


def create_animation(namespace, params, frames, fps):
    fig_cfg = get_figure_config(namespace)
    state = call_init_state(namespace["init_state"], params)
    fig, ax = make_figure(fig_cfg)

    update_state = namespace["update_state"]
    draw_frame = namespace["draw_frame"]

    def _update(frame):
        nonlocal state
        state = call_update_state(update_state, state, params, frame)
        call_draw_frame(draw_frame, ax, state, params, frame)
        return []

    ani = FuncAnimation(
        fig,
        _update,
        frames=int(frames),
        interval=1000 / max(int(fps), 1),
        blit=False
    )
    return ani, fig


def make_gpt_prompt(topic, visual_goal, extra_condition):
    # IMPORTANT: double braces are used so this f-string can include literal JSON/dict examples.
    return f"""내가 만들고 싶은 시뮬레이션 주제는 [{topic}]이야.

보여주고 싶은 모습:
{visual_goal}

추가 조건:
{extra_condition}

Python으로 파라미터 조절 가능한 애니메이션을 만들고 싶어.
나는 네가 작성한 코드를 웹사이트에 붙여넣어서 실행할 거야.
어떤 주제가 올지 모르기 때문에, 파라미터와 시각화 방식은 네가 주제에 맞게 직접 정해야 해.

아래 4개는 반드시 작성해줘.

1. PARAM_CONFIG
2. init_state(params)
3. update_state(state, params, frame)
4. draw_frame(ax, state, params, frame)

필요하면 선택적으로 아래도 작성해줘.

5. FIGURE_CONFIG

[PARAM_CONFIG 작성 규칙]
- 사용자가 조절하면 좋은 파라미터를 주제에 맞게 직접 선정해줘.
- 각 파라미터는 dictionary로 작성해줘.
- 가능한 type은 "slider", "number", "int", "checkbox", "color", "text", "select", "log_slider" 중에서 골라줘.
- slider/number/int/log_slider에는 가능한 한 min, max, default, step을 넣어줘.
- select에는 options와 default를 넣어줘.
- 모든 파라미터에는 description을 넣어줘.
- 단위가 있으면 unit도 넣어줘.

[함수 작성 규칙]
- init_state(params)는 초기 상태를 만들고 state dictionary를 return해야 해.
- update_state(state, params, frame)는 프레임마다 상태를 업데이트하고 state를 return해야 해.
- draw_frame(ax, state, params, frame)는 matplotlib의 ax 객체에 직접 그림을 그려야 해.
- draw_frame 안에서 축 이름, 제목, 범례, 색상, grid 등을 주제에 맞게 직접 정해줘.
- Streamlit 코드, 저장 코드, FuncAnimation 코드는 쓰지 마.
- numpy는 np로 이미 import되어 있다고 가정해.
- matplotlib.pyplot은 쓰지 말고 ax 중심으로 그려줘.
- 파일 입출력은 하지 마.
- 외부 라이브러리는 사용하지 마.

[FIGURE_CONFIG 작성 규칙]
- 2D 그래프면 {{"projection": "2d"}}로 설정해줘.
- 3D 그래프가 필요하면 {{"projection": "3d"}}로 설정해줘.
- 예: FIGURE_CONFIG = {{"projection": "2d", "figsize": (8, 5.2), "dpi": 110}}

[최종 출력 형식]
먼저 파라미터를 왜 그렇게 골랐는지 간단히 설명해줘.
그 다음 마지막에는 복사하기 쉽게 Python 코드블록 하나로 PARAM_CONFIG, FIGURE_CONFIG, init_state, update_state, draw_frame을 모두 정리해줘.
"""


# =========================================================
# Sidebar: app/global controls
# =========================================================
st.sidebar.markdown("## 🧩 App Controls")

file_name = sanitize_filename(
    st.sidebar.text_input("file name prefix", value="simulation")
)

frames = st.sidebar.number_input(
    "final GIF frames",
    min_value=5,
    max_value=1000,
    value=160,
    step=5,
    help="GIF로 렌더링할 프레임 수"
)

fps = st.sidebar.number_input(
    "GIF fps",
    min_value=1,
    max_value=60,
    value=20,
    step=1,
    help="GIF 초당 프레임 수"
)

preview_frame = st.sidebar.number_input(
    "preview frame",
    min_value=0,
    max_value=1000,
    value=40,
    step=5,
    help="Preview에서 몇 번째 프레임을 보여줄지"
)

st.sidebar.markdown("---")

# =========================================================
# Tabs
# =========================================================
tab_prompt, tab_code, tab_params, tab_preview, tab_download, tab_help = st.tabs(
    ["① GPT Prompt", "② Code Paste", "③ Parameters", "④ Preview", "⑤ Download", "⑥ Help"]
)

# =========================================================
# Tab 1: Prompt generator
# =========================================================
with tab_prompt:
    st.markdown("### 🪄 GPT에게 줄 요청문 만들기")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        topic = st.text_input(
            "시뮬레이션 주제",
            value="강유전체에서의 분극"
        )

        visual_goal = st.text_area(
            "보여주고 싶은 모습",
            value="분극이 되는 모습과 온도에 따라 속도 변화와 분극 크기가 달라지는 모습을 보고 싶어.",
            height=110
        )

    with col_b:
        extra_condition = st.text_area(
            "추가 조건",
            value="파라미터를 조절하면 그래프 모양이 확실히 달라져야 해. 발표용이므로 축 이름, 범례, 제목을 깔끔하게 넣어줘.",
            height=162
        )

    generated_prompt = make_gpt_prompt(topic, visual_goal, extra_condition)

    st.markdown("아래 프롬프트를 복사해서 GPT에게 붙여넣으면 됩니다.")
    st.code(generated_prompt, language="text")

    st.download_button(
        "📥 Download Prompt",
        data=generated_prompt.encode("utf-8"),
        file_name=f"{file_name}_{now_stamp()}_prompt.txt",
        mime="text/plain",
        use_container_width=True
    )

# =========================================================
# Tab 2: Code paste / execute
# =========================================================
with tab_code:
    st.markdown("### 🧬 GPT가 만든 코드 붙여넣기")

    st.markdown(
        """
<div class="small-muted">
GPT 답변 전체를 붙여넣어도 됩니다. 앱이 Python 코드블록을 자동으로 추출합니다.
필수 구조는 <code>PARAM_CONFIG</code>, <code>init_state</code>, <code>update_state</code>, <code>draw_frame</code>입니다.
</div>
""",
        unsafe_allow_html=True
    )

    raw_code = st.text_area(
        "Paste GPT answer or Python code here",
        value=get_default_code(),
        height=520,
        key="raw_code_area"
    )

    user_code = extract_python_code(raw_code)

    with st.expander("🔍 실제 실행될 코드 확인", expanded=False):
        st.code(user_code, language="python")

# Execute code globally after code tab has been declared.
if "raw_code" not in locals():
    raw_code = get_default_code()

user_code = extract_python_code(raw_code)

try:
    namespace = execute_user_code(user_code)
    validation_errors = validate_namespace(namespace)
    ok = len(validation_errors) == 0
    exec_error = None
except Exception:
    namespace = {}
    validation_errors = []
    ok = False
    exec_error = traceback.format_exc()

# =========================================================
# Tab 3 + Sidebar dynamic parameter widgets
# =========================================================
st.sidebar.markdown("## 🎛️ Parameters")

params = {}
param_config = namespace.get("PARAM_CONFIG", {}) if ok else {}

if ok:
    if not param_config:
        st.sidebar.warning("PARAM_CONFIG가 비어 있습니다.")
    else:
        for pname, pcfg in param_config.items():
            params[pname] = create_param_widget(pname, pcfg, key_prefix="dynamic")
else:
    st.sidebar.info("코드가 정상 인식되면 파라미터 조절창이 여기에 생성됩니다.")

with tab_params:
    st.markdown("### 🎛️ 감지된 파라미터")

    if exec_error:
        st.markdown('<div class="danger-box"><b>코드 실행 오류가 있습니다.</b></div>', unsafe_allow_html=True)
        st.code(exec_error, language="text")
    elif not ok:
        st.markdown('<div class="danger-box"><b>필수 구조가 부족합니다.</b></div>', unsafe_allow_html=True)
        st.write(validation_errors)
    else:
        st.markdown('<div class="ok-box"><b>코드 인식 성공!</b> 왼쪽 사이드바에서 파라미터를 조절할 수 있습니다.</div>', unsafe_allow_html=True)

        fig_cfg = get_figure_config(namespace)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parameters", len(param_config))
        c2.metric("Projection", fig_cfg["projection"])
        c3.metric("Figsize", str(fig_cfg["figsize"]))
        c4.metric("DPI", fig_cfg["dpi"])

        st.markdown("#### PARAM_CONFIG")
        st.json(json_safe(param_config))

        st.markdown("#### Current parameter values")
        st.json(json_safe(params))

# =========================================================
# Tab 4: Preview
# =========================================================
with tab_preview:
    st.markdown("### 👀 Preview")

    if not ok:
        st.error("코드가 아직 준비되지 않았습니다. ② Code Paste 탭의 코드를 확인하세요.")
        if exec_error:
            st.code(exec_error, language="text")
        elif validation_errors:
            st.write(validation_errors)
    else:
        st.markdown(
            """
<div class="cute-card">
왼쪽 사이드바에서 파라미터를 조절한 뒤 <b>Update Preview</b>를 누르세요.
GIF 저장 전 빠르게 모양을 확인하는 단계입니다.
</div>
""",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            update_preview = st.button("🔄 Update Preview", use_container_width=True)

        with col2:
            run_smoke = st.button("🧪 Run Smoke Test", use_container_width=True)

        with col3:
            st.metric("Preview frame", int(preview_frame))

        if run_smoke:
            try:
                test_state = call_init_state(namespace["init_state"], params)
                for f in range(3):
                    test_state = call_update_state(namespace["update_state"], test_state, params, f)
                fig = render_single_frame(namespace, params, 3)
                plt.close(fig)
                st.success("Smoke test 통과: init/update/draw가 정상 실행되었습니다.")
            except Exception:
                st.error("Smoke test 실패")
                st.code(traceback.format_exc(), language="text")

        if update_preview:
            try:
                fig = render_single_frame(namespace, params, int(preview_frame))
                st.pyplot(fig)
                plt.close(fig)
            except Exception:
                st.error("Preview 렌더링 오류")
                st.code(traceback.format_exc(), language="text")

# =========================================================
# Tab 5: Render and downloads
# =========================================================
with tab_download:
    st.markdown("### 📦 Render & Download")

    st.markdown(
        """
<div class="warn-box">
웹사이트 배포 버전에서는 파일이 내 컴퓨터 폴더에 바로 저장되지 않습니다.
먼저 Render 버튼으로 서버 메모리에 생성한 뒤, Download 버튼으로 내려받으세요.
</div>
""",
        unsafe_allow_html=True
    )

    if "generated_gif_bytes" not in st.session_state:
        st.session_state.generated_gif_bytes = None
    if "generated_gif_name" not in st.session_state:
        st.session_state.generated_gif_name = None
    if "generated_png_bytes" not in st.session_state:
        st.session_state.generated_png_bytes = None
    if "generated_png_name" not in st.session_state:
        st.session_state.generated_png_name = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎬 Render GIF", use_container_width=True):
            if not ok:
                st.error("코드가 준비되지 않아 GIF를 만들 수 없습니다.")
            else:
                try:
                    with st.spinner("GIF 생성 중입니다. 프레임 수가 많으면 시간이 걸릴 수 있습니다."):
                        with tempfile.TemporaryDirectory() as tmpdir:
                            gif_name = f"{file_name}_{now_stamp()}.gif"
                            gif_path = Path(tmpdir) / gif_name

                            ani, fig = create_animation(namespace, params, int(frames), int(fps))
                            ani.save(gif_path, writer=PillowWriter(fps=int(fps)))
                            plt.close(fig)

                            st.session_state.generated_gif_bytes = gif_path.read_bytes()
                            st.session_state.generated_gif_name = gif_name

                    st.success("GIF 생성 완료! 아래 버튼으로 다운로드하세요.")
                except Exception:
                    st.error("GIF 생성 오류")
                    st.code(traceback.format_exc(), language="text")

        if st.session_state.generated_gif_bytes is not None:
            st.download_button(
                label="📥 Download GIF",
                data=st.session_state.generated_gif_bytes,
                file_name=st.session_state.generated_gif_name,
                mime="image/gif",
                use_container_width=True
            )

    with col2:
        if st.button("🖼️ Render PNG Preview", use_container_width=True):
            if not ok:
                st.error("코드가 준비되지 않아 PNG를 만들 수 없습니다.")
            else:
                try:
                    fig_cfg = get_figure_config(namespace)
                    fig = render_single_frame(namespace, params, int(preview_frame))

                    buffer = io.BytesIO()
                    png_name = f"{file_name}_{now_stamp()}_preview.png"
                    fig.savefig(buffer, format="png", dpi=int(fig_cfg["dpi"]), bbox_inches="tight")
                    plt.close(fig)
                    buffer.seek(0)

                    st.session_state.generated_png_bytes = buffer.getvalue()
                    st.session_state.generated_png_name = png_name

                    st.success("PNG 생성 완료! 아래 버튼으로 다운로드하세요.")
                except Exception:
                    st.error("PNG 생성 오류")
                    st.code(traceback.format_exc(), language="text")

        if st.session_state.generated_png_bytes is not None:
            st.download_button(
                label="📥 Download PNG",
                data=st.session_state.generated_png_bytes,
                file_name=st.session_state.generated_png_name,
                mime="image/png",
                use_container_width=True
            )

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.download_button(
            label="🐍 Download Python Code",
            data=user_code.encode("utf-8"),
            file_name=f"{file_name}_{now_stamp()}_code.py",
            mime="text/x-python",
            use_container_width=True
        )

    with col4:
        params_json = json.dumps(json_safe(params), indent=4, ensure_ascii=False)
        st.download_button(
            label="🧾 Download Parameters JSON",
            data=params_json.encode("utf-8"),
            file_name=f"{file_name}_{now_stamp()}_params.json",
            mime="application/json",
            use_container_width=True
        )

# =========================================================
# Tab 6: Help
# =========================================================
with tab_help:
    st.markdown("### 🧭 사용 설명")

    st.markdown(
        """
<div class="cute-card">
<b>사용 흐름</b><br>
1. ① GPT Prompt에서 주제와 원하는 모습을 입력합니다.<br>
2. 생성된 프롬프트를 GPT에게 보냅니다.<br>
3. GPT가 만든 코드블록을 ② Code Paste에 붙여넣습니다.<br>
4. ③ Parameters에서 자동 생성된 파라미터를 조절합니다.<br>
5. ④ Preview에서 확인합니다.<br>
6. ⑤ Download에서 GIF, PNG, 코드, 파라미터를 내려받습니다.
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("#### GPT가 작성해야 하는 기본 구조")

    st.code(
        """
PARAM_CONFIG = {
    "parameter_name": {
        "type": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "description": "What this parameter controls."
    }
}

FIGURE_CONFIG = {
    "projection": "2d",
    "figsize": (8, 5.2),
    "dpi": 110
}

def init_state(params):
    state = {}
    return state

def update_state(state, params, frame):
    return state

def draw_frame(ax, state, params, frame):
    ax.clear()
    ax.set_title("Simulation")
""",
        language="python"
    )

    st.markdown("#### 지원하는 PARAM_CONFIG type")

    st.markdown(
        """
- `slider`: 범위가 있는 실수 슬라이더
- `number`: 직접 입력하는 실수
- `int`: 정수 입력
- `checkbox`: True/False
- `color`: 색상 선택
- `text`: 문자열 입력
- `select`: 선택 메뉴
- `log_slider`: 로그 스케일 슬라이더
"""
    )

    st.markdown("#### 배포 버전 저장 방식")

    st.info(
        "Streamlit Cloud에서는 서버 내부 경로에 저장된 파일을 사용자가 직접 찾을 수 없습니다. 그래서 이 앱은 Render 후 Download 버튼으로 파일을 받는 구조입니다."
    )

    st.warning(
        "주의: 이 앱은 붙여넣은 Python 코드를 실행합니다. 신뢰할 수 없는 사용자가 임의 코드를 실행하지 않도록 공개 범위를 조심하세요."
    )
