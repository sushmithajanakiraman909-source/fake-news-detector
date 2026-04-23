"""
Fake News & AI Content Detector
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Fake News & AI Content Detector",
    page_icon="🔍",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8ecf4;
}
.stApp { background: #0d0f14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; }

/* Hero */
.hero {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    border-bottom: 1px solid #2a3040;
    margin-bottom: 1.5rem;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,229,255,0.06) 0%, transparent 65%);
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #e8ecf4;
    letter-spacing: -1px;
    margin: 0 0 0.3rem;
}
.hero h1 span { color: #00e5ff; }
.hero p { color: #6b7690; font-size: 0.9rem; margin: 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161a23;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #2a3040;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #6b7690;
    background: transparent;
    border-radius: 7px;
    padding: 0.55rem 1.2rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1e2433 !important;
    color: #00e5ff !important;
}

/* Text area */
.stTextArea textarea {
    background: #161a23 !important;
    border: 1px solid #2a3040 !important;
    color: #e8ecf4 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
}
.stTextArea textarea:focus {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.08) !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #00e5ff !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    width: 100%;
}
.stButton > button:hover {
    background: #33eaff !important;
    box-shadow: 0 4px 18px rgba(0,229,255,0.22) !important;
}

/* Cards */
.rcard {
    background: #161a23;
    border: 1px solid #2a3040;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.6rem 0;
    position: relative;
    overflow: hidden;
}
.rcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
}
.rcard.fake::before  { background: #ff4b6e; }
.rcard.real::before  { background: #00c98d; }
.rcard.ai::before    { background: #a259ff; }
.rcard.human::before { background: #00e5ff; }
.rcard.neutral::before { background: #6b7690; }

.rlabel {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6b7690;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.25rem;
}
.rvalue {
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 0.15rem;
}
.rvalue.fake   { color: #ff4b6e; }
.rvalue.real   { color: #00c98d; }
.rvalue.ai     { color: #a259ff; }
.rvalue.human  { color: #00e5ff; }
.rvalue.neutral { color: #e8ecf4; }
.rsub { font-size: 0.8rem; color: #6b7690; }

/* Confidence bar */
.cbar-wrap { margin: 0.5rem 0 0.2rem; }
.cbar-top {
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #6b7690;
    margin-bottom: 0.25rem;
}
.cbar-bg {
    background: #1e2433;
    border-radius: 100px;
    height: 7px;
    overflow: hidden;
}
.cbar-fill { height: 100%; border-radius: 100px; }
.cbar-fill.fake  { background: linear-gradient(90deg,#ff4b6e,#ff6b8a); }
.cbar-fill.real  { background: linear-gradient(90deg,#00c98d,#00e5b8); }
.cbar-fill.ai    { background: linear-gradient(90deg,#a259ff,#bf7fff); }
.cbar-fill.human { background: linear-gradient(90deg,#00e5ff,#66f0ff); }

/* Metric cells */
.mcell {
    background: #1e2433;
    border: 1px solid #2a3040;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin: 0.3rem 0;
}
.mcell .ml {
    font-family:'Space Mono',monospace;
    font-size:0.62rem;
    color:#6b7690;
    text-transform:uppercase;
    letter-spacing:1px;
}
.mcell .mv { font-size:1rem; font-weight:600; color:#e8ecf4; margin-top:0.15rem; }

/* Section head */
.shead {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6b7690;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #2a3040;
    margin: 1.2rem 0 0.7rem;
}

/* Dots */
.dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot.g { background:#00c98d; box-shadow:0 0 5px #00c98d; }
.dot.r { background:#ff4b6e; box-shadow:0 0 5px #ff4b6e; }
.dot.p { background:#a259ff; box-shadow:0 0 5px #a259ff; }
.dot.c { background:#00e5ff; box-shadow:0 0 5px #00e5ff; }
.dot.m { background:#6b7690; }

/* Info box */
.ibox {
    background: rgba(0,229,255,0.04);
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #6b7690;
}
.ibox-center { text-align:center; padding: 2.5rem 1rem; }
.ibox-center .ico { font-size: 2.2rem; margin-bottom: 0.4rem; }

/* Disclaimer */
.disc {
    background: rgba(255,75,110,0.06);
    border: 1px solid rgba(255,75,110,0.18);
    border-radius: 8px;
    padding: 0.7rem 1.2rem;
    font-size: 0.78rem;
    color: #6b7690;
    text-align: center;
    font-family: 'Space Mono', monospace;
    margin-top: 1.5rem;
}
.disc span { color: #ff4b6e; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 Fake News &amp; <span>AI Content</span> Detector</h1>
  <p>Prototype · Text &amp; Image Analysis · sklearn + OpenCV</p>
</div>
""", unsafe_allow_html=True)


# ── Build / cache ML model ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training classifier...")
def build_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    fake = [
        "SHOCKING: Scientists prove drinking bleach cures all diseases instantly!",
        "BREAKING: Government secretly chips COVID vaccines with mind-control nanobots!",
        "You won't believe what they're hiding: 5G towers cause cancer and media won't report it!",
        "EXPOSED: Bill Gates funds secret population control through chemtrails!",
        "MIRACLE CURE: Ancient remedy big pharma doesn't want you to know destroys cancer in 24 hours!",
        "ALERT: Deep state plans to replace the sun with an artificial light source in 2025!",
        "MUST SHARE: Fluoride in water supply linked to IQ reduction, government admits in leaked doc!",
        "Doctors HATE this one weird trick that eliminates diabetes forever with zero side effects!",
        "URGENT: Eating vegetables causes autism, suppressed study finds, share before deleted!",
        "The moon landing was 100% faked in a Hollywood studio, new evidence proves it completely!",
        "Coronavirus was engineered in a secret lab and released deliberately to start World War 3!",
        "Banks plan to steal your savings overnight, withdraw cash NOW before it is too late!!",
        "Scientists who discovered flat earth truth are being silenced and killed by elites!",
        "EXCLUSIVE: Aliens already living among us in disguise, government knows and hides it!",
        "SHOCKING TRUTH: Sunscreen causes skin cancer, leaked study reveals the terrible truth!",
        "Democrats planning to cancel the next election and install permanent dictatorship in 2025!",
        "New secret law will allow government to enter your home without warrant starting next month!",
        "Drinking hot water with lemon DESTROYS all cancer cells, oncologists will never tell you!",
        "Secret underground cities built for elites to survive coming asteroid impact this year!",
        "WiFi radiation causes mass infertility in children, tech giants buried the shocking study!",
        "Global warming is a total hoax created by China to destroy American manufacturing forever!",
        "BREAKING: Major city to be hit by massive terrorist attack this weekend, FBI source says!",
        "Scientists PROVE prayer can physically change DNA and cure all genetic diseases overnight!",
        "Mainstream media is completely controlled by a secret globalist cabal suppressing the truth!",
        "100% PROOF vaccines contain aborted fetal cells used to program obedient children globally!",
        "The earth is actually flat and NASA has been lying about it for decades to steal your money!",
        "Hollywood celebrities drink children's blood to stay young, leaked documents expose everything!",
        "New world order meeting EXPOSED: elites plan massive population reduction by end of year!",
        "SHARE NOW: This natural plant kills 98% of cancer cells in 16 hours and pharma is furious!",
        "George Soros caught on video admitting he funds all major protests to destroy America!!",
    ]

    real = [
        "Scientists discovered a new species of deep-sea fish near the Mariana Trench this month.",
        "The Federal Reserve announced a quarter-point interest rate increase on Wednesday afternoon.",
        "A study in Nature found that regular exercise significantly reduces the risk of heart disease.",
        "The United Nations climate summit concluded with agreements from more than 120 countries.",
        "Researchers at MIT developed a new battery technology capable of charging in five minutes.",
        "The S&P 500 closed higher on strong quarterly earnings reports from major technology firms.",
        "Health officials confirmed a new measles case in the county and are urging residents to vaccinate.",
        "The city council voted 7-2 to approve the new public transit expansion plan on Thursday.",
        "Astronomers detected gravitational waves from a collision between two neutron stars last week.",
        "A bipartisan infrastructure bill passed the Senate and now moves to the House for a vote.",
        "The World Health Organization updated its guidelines on antibiotic use to curb drug resistance.",
        "Electric vehicle sales rose 24 percent in the first quarter compared to the same period last year.",
        "The Supreme Court agreed to hear arguments on the landmark water rights environmental case.",
        "Tropical storm warnings issued for coastal regions as the system gains strength in the Atlantic.",
        "New archaeological findings suggest ancient trade routes were significantly broader than believed.",
        "The unemployment rate fell to 3.8 percent according to the latest Bureau of Labor Statistics report.",
        "A clinical trial showed the new blood pressure medication reduces readings significantly in six weeks.",
        "Firefighters contained the wildfire to 4,000 acres after three days of battling difficult terrain.",
        "The president signed an executive order on consumer data privacy protections on Friday.",
        "The local school district reports improved graduation rates following new curriculum implementation.",
        "The International Space Station crew completed a six-hour spacewalk to repair damaged solar panels.",
        "The central bank held interest rates steady amid concerns about slowing global economic growth.",
        "Researchers published findings linking elevated air pollution to increased respiratory illness rates.",
        "The company announced plans to hire 2,000 new workers and open a manufacturing plant in Ohio.",
        "Government data shows a measurable decline in violent crime rates across most major US cities.",
        "The European Union approved new regulations on artificial intelligence applications in member states.",
        "Scientists confirmed that the James Webb telescope captured images of galaxies from 13 billion years ago.",
        "The pharmaceutical company reported successful phase three trial results for its new diabetes drug.",
        "A new report from the Congressional Budget Office projects a smaller deficit than previously forecast.",
        "The trade agreement between the two nations was finalized after eighteen months of negotiations.",
    ]

    texts  = fake + real
    labels = [1]*len(fake) + [0]*len(real)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(C=1.5, max_iter=1000, random_state=42)),
    ])
    pipe.fit(texts, labels)
    return pipe


model = build_model()


# ── Text analysis helpers ────────────────────────────────────────────────────
def predict_fake(text):
    p      = model.predict_proba([text])[0]
    fake_p = float(p[1]) * 100
    real_p = float(p[0]) * 100
    label  = "Fake News" if fake_p > 50 else "Real News"
    conf   = max(fake_p, real_p)
    return label, conf, fake_p, real_p


AI_PHRASES = [
    "it is important to note", "it is worth noting", "in conclusion",
    "furthermore", "moreover", "in summary", "it should be noted",
    "as mentioned above", "in addition", "on the other hand",
    "to summarize", "needless to say", "it goes without saying",
    "with that being said", "delve into", "commendable", "invaluable",
    "it is crucial", "in the realm of", "foster", "facilitate",
    "leverage", "utilize", "paramount", "intricate", "pivotal",
    "having said that", "this is because", "as a result",
    "comprehensive", "streamline", "revolutionize", "transformative",
]

CONTRACTIONS = re.compile(
    r"\b(don't|can't|won't|I'm|it's|we're|they're|isn't|aren't|didn't|"
    r"couldn't|wouldn't|I've|you've|we've|they've|I'd|he'd|she'd|they'd|"
    r"I'll|you'll|we'll|they'll|that's|what's|here's|there's|who's|"
    r"how's|when's|let's|shan't)\b",
    re.IGNORECASE
)


def analyze_ai(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 3]
    words     = re.findall(r'\b\w+\b', text.lower())

    if len(words) < 15 or not sentences:
        return "Too short to analyze", 0.0, 100.0, {}, "neutral"

    sent_lens    = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    variance     = float(np.var(sent_lens)) if len(sent_lens) > 1 else 0.0
    avg_len      = float(np.mean(sent_lens))
    ttr          = len(set(words)) / len(words)
    contractions = len(CONTRACTIONS.findall(text))
    ai_phrase_ct = sum(1 for p in AI_PHRASES if p in text.lower())
    exclamations = text.count('!') + text.count('?')
    first_words  = [s.split()[0].lower() for s in sentences if s.split()]
    start_div    = len(set(first_words)) / len(first_words) if first_words else 1.0

    score  = 0
    score += 2 if variance < 15  else (1 if variance < 40  else 0)
    score += 2 if ttr < 0.5      else (1 if ttr < 0.65     else 0)
    score += 2 if (contractions == 0 and len(words) > 40) else (1 if contractions <= 1 else 0)
    score += min(ai_phrase_ct * 2, 4)
    score += 1 if (exclamations == 0 and len(words) > 60) else 0
    score += 1 if start_div < 0.5 else 0

    ai_p   = min(score / 12.0, 1.0) * 100
    hum_p  = 100.0 - ai_p

    verdict = (
        "AI Generated: Likely"   if ai_p >= 55 else
        "AI Generated: Possible" if ai_p >= 35 else
        "Human Written: Likely"
    )
    cls = "ai" if ai_p >= 35 else "human"

    metrics = {
        "Avg Sent Length"  : f"{avg_len:.1f} words",
        "Length Variance"  : f"{variance:.1f}",
        "Lexical Diversity": f"{ttr:.2f}",
        "Contractions"     : str(contractions),
        "AI Phrases Found" : str(ai_phrase_ct),
        "Start Diversity"  : f"{start_div:.2f}",
    }
    return verdict, ai_p, hum_p, metrics, cls


# ── Image analysis helpers ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading face detector...")
def load_cv2():
    try:
        import cv2
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return cv2, cascade
    except Exception:
        return None, None


def analyze_image(uploaded):
    from PIL import Image
    cv2, cascade = load_cv2()

    img = Image.open(uploaded).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    # Face detection
    face_count = 0
    face_ok    = False
    if cv2 is not None and cascade is not None and not cascade.empty():
        gray       = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        faces      = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        face_count = len(faces)
        face_ok    = True

    # AI-image heuristics
    ch_stds = [float(np.std(arr[:, :, c])) for c in range(3)]
    avg_std = float(np.mean(ch_stds))

    gray_f  = arr.mean(axis=2)
    lap     = gray_f[1:-1, 1:-1] - (
                  gray_f[:-2, 1:-1] + gray_f[2:, 1:-1] +
                  gray_f[1:-1, :-2] + gray_f[1:-1, 2:]
              ) / 4
    lap_var = float(np.var(np.abs(lap)))

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mx      = np.maximum(np.maximum(r, g), b)
    mn      = np.minimum(np.minimum(r, g), b)
    sat     = np.where(mx > 0, (mx - mn) / (mx + 1e-5), 0)
    sat_std = float(np.std(sat))

    s  = 0
    s += 2 if avg_std < 40  else (1 if avg_std < 55  else 0)
    s += 2 if lap_var < 30  else (1 if lap_var < 60  else 0)
    s += 1 if sat_std < 0.08 else 0

    ai_p   = min(s / 5.0, 1.0) * 100
    real_p = 100.0 - ai_p

    ai_verdict = "Possibly AI Generated" if ai_p >= 60 else "Likely Real Image"
    ai_cls     = "ai" if ai_p >= 60 else "human"

    return {
        "face_ok"   : face_ok,
        "face_count": face_count,
        "ai_verdict": ai_verdict,
        "ai_cls"    : ai_cls,
        "ai_pct"    : ai_p,
        "real_pct"  : real_p,
        "size"      : f"{img.width} x {img.height}",
        "mode"      : img.mode,
        "metrics"   : {
            "Color Std Dev"   : f"{avg_std:.1f}",
            "Texture Variance": f"{lap_var:.1f}",
            "Saturation Std"  : f"{sat_std:.3f}",
        },
    }, img


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝  Text Analysis", "🖼️  Image Analysis"])


# ══════════════════════════════════════════════
#  TAB 1 – TEXT
# ══════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="shead">Input Text</div>', unsafe_allow_html=True)

        samples = {
            "-- Quick sample --": "",
            "Real News sample": (
                "The Federal Reserve kept its benchmark interest rate unchanged on Wednesday, "
                "citing continued progress on inflation and a resilient labor market. Fed Chair "
                "Jerome Powell said policymakers remain data-dependent as they evaluate future moves."
            ),
            "Fake News sample": (
                "BREAKING: Scientists have PROVEN that drinking apple cider vinegar every morning "
                "DESTROYS cancer cells in 24 hours! Big pharma doesn't want you to know this ancient "
                "secret. SHARE before they delete this shocking truth!!"
            ),
            "AI-style text": (
                "It is important to note that artificial intelligence has fundamentally transformed "
                "various sectors of modern society. Furthermore, the implications of these technological "
                "advancements are multifaceted and require careful consideration. Moreover, stakeholders "
                "must leverage these innovations to facilitate optimal outcomes. In conclusion, the "
                "paramount significance of AI cannot be overstated in the current realm of discourse."
            ),
        }

        choice  = st.selectbox("Load a sample:", list(samples.keys()))
        default = samples[choice]
        txt     = st.text_area(
            "Paste your text here:",
            value=default,
            height=210,
            placeholder="Enter news article, social post, or any text...",
        )

        wc = len(txt.split()) if txt.strip() else 0
        st.markdown(
            f'<div class="ibox">Word count: {wc} &nbsp;|&nbsp; Characters: {len(txt)}</div>',
            unsafe_allow_html=True,
        )
        go = st.button("Analyze Text", key="btn_text")

    with right:
        st.markdown('<div class="shead">Results</div>', unsafe_allow_html=True)

        if go:
            if len(txt.strip()) < 20:
                st.warning("Please enter at least 20 characters.")
            else:
                with st.spinner("Analyzing..."):
                    # Fake news prediction
                    label, conf, fake_p, real_p = predict_fake(txt)
                    is_fake = label == "Fake News"
                    cc      = "fake" if is_fake else "real"
                    dot_c   = "r" if is_fake else "g"
                    icon    = "Warning" if is_fake else "OK"

                    st.markdown(f"""
                    <div class="rcard {cc}">
                      <div class="rlabel">Fake News Detection</div>
                      <div class="rvalue {cc}">
                        <span class="dot {dot_c}"></span>{icon}: {label}
                      </div>
                      <div class="cbar-wrap">
                        <div class="cbar-top">
                          <span>Confidence</span><span>{conf:.1f}%</span>
                        </div>
                        <div class="cbar-bg">
                          <div class="cbar-fill {cc}" style="width:{conf:.1f}%"></div>
                        </div>
                      </div>
                      <div class="rsub">Fake: {fake_p:.1f}% &nbsp;|&nbsp; Real: {real_p:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                    # AI vs Human
                    verdict, ai_p, hum_p, ai_m, ai_cls = analyze_ai(txt)
                    bar_pct = ai_p if ai_cls == "ai" else hum_p
                    bar_cls = ai_cls if ai_cls in ("ai", "human") else "human"
                    a_dot   = "p" if ai_cls == "ai" else "c"

                    st.markdown(f"""
                    <div class="rcard {ai_cls}">
                      <div class="rlabel">AI vs Human Detection</div>
                      <div class="rvalue {ai_cls}">
                        <span class="dot {a_dot}"></span>{verdict}
                      </div>
                      <div class="cbar-wrap">
                        <div class="cbar-top">
                          <span>{'AI likelihood' if ai_cls=='ai' else 'Human likelihood'}</span>
                          <span>{bar_pct:.1f}%</span>
                        </div>
                        <div class="cbar-bg">
                          <div class="cbar-fill {bar_cls}" style="width:{bar_pct:.1f}%"></div>
                        </div>
                      </div>
                      <div class="rsub">AI score: {ai_p:.1f}% &nbsp;|&nbsp; Human score: {hum_p:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                    # Metrics
                    if ai_m:
                        st.markdown('<div class="shead">Text Heuristics</div>', unsafe_allow_html=True)
                        cols = st.columns(3)
                        for i, (k, v) in enumerate(ai_m.items()):
                            with cols[i % 3]:
                                st.markdown(
                                    f'<div class="mcell"><div class="ml">{k}</div>'
                                    f'<div class="mv">{v}</div></div>',
                                    unsafe_allow_html=True,
                                )
        else:
            st.markdown("""
            <div class="ibox ibox-center">
              <div class="ico">📋</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#6b7690;">
                Enter text then click Analyze Text
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 2 – IMAGE
# ══════════════════════════════════════════════
with tab2:
    l2, r2 = st.columns([1, 1], gap="large")

    with l2:
        st.markdown('<div class="shead">Upload Image</div>', unsafe_allow_html=True)
        ufile = st.file_uploader(
            "Choose JPG / PNG / WEBP",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )
        if ufile:
            st.image(ufile, caption="Uploaded image", use_container_width=True)
            go_img = st.button("Analyze Image", key="btn_img")
        else:
            st.markdown("""
            <div class="ibox ibox-center">
              <div class="ico">🖼️</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#6b7690;">
                Drag and drop or browse<br>JPG · PNG · WEBP · BMP
              </div>
            </div>""", unsafe_allow_html=True)
            go_img = False

    with r2:
        st.markdown('<div class="shead">Image Analysis Results</div>', unsafe_allow_html=True)

        if ufile and go_img:
            with st.spinner("Analyzing image..."):
                try:
                    res, _ = analyze_image(ufile)

                    # Face card
                    fok = res["face_ok"]
                    fc  = res["face_count"]
                    if fok and fc > 0:
                        f_label = f"Yes — {fc} face{'s' if fc != 1 else ''} detected"
                        f_cls   = "real"
                        f_dot   = "g"
                    elif fok:
                        f_label = "No faces detected"
                        f_cls   = "neutral"
                        f_dot   = "m"
                    else:
                        f_label = "OpenCV unavailable"
                        f_cls   = "neutral"
                        f_dot   = "m"

                    st.markdown(f"""
                    <div class="rcard {f_cls}">
                      <div class="rlabel">Face Detection</div>
                      <div class="rvalue {f_cls}">
                        <span class="dot {f_dot}"></span>Face Detected: {f_label}
                      </div>
                      <div class="rsub">
                        {'Human face(s) present in the image.' if (fok and fc > 0) else 'No human faces found.'}
                      </div>
                    </div>""", unsafe_allow_html=True)

                    # AI image card
                    av  = res["ai_verdict"]
                    ac  = res["ai_cls"]
                    ap  = res["ai_pct"]
                    rp  = res["real_pct"]
                    a_d = "p" if ac == "ai" else "c"
                    bp  = ap if ac == "ai" else rp

                    st.markdown(f"""
                    <div class="rcard {ac}">
                      <div class="rlabel">Image Authenticity</div>
                      <div class="rvalue {ac}">
                        <span class="dot {a_d}"></span>{av}
                      </div>
                      <div class="cbar-wrap">
                        <div class="cbar-top">
                          <span>{'AI likelihood' if ac=='ai' else 'Real likelihood'}</span>
                          <span>{bp:.1f}%</span>
                        </div>
                        <div class="cbar-bg">
                          <div class="cbar-fill {ac}" style="width:{bp:.1f}%"></div>
                        </div>
                      </div>
                      <div class="rsub">AI: {ap:.1f}% &nbsp;|&nbsp; Real: {rp:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                    # Metrics
                    st.markdown('<div class="shead">Image Metrics</div>', unsafe_allow_html=True)
                    all_m = {
                        "Dimensions": res["size"],
                        "Color Mode": res["mode"],
                        **res["metrics"],
                    }
                    cols3 = st.columns(3)
                    for i, (k, v) in enumerate(all_m.items()):
                        with cols3[i % 3]:
                            st.markdown(
                                f'<div class="mcell"><div class="ml">{k}</div>'
                                f'<div class="mv">{v}</div></div>',
                                unsafe_allow_html=True,
                            )

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Ensure Pillow and OpenCV are installed:\n"
                            "`pip install pillow opencv-python-headless`")

        elif not ufile:
            st.markdown("""
            <div class="ibox ibox-center">
              <div class="ico">📊</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#6b7690;">
                Upload an image then click Analyze Image
              </div>
            </div>""", unsafe_allow_html=True)


# ── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="disc">
  <span>DISCLAIMER:</span> This is a prototype.
  Results may not be 100% accurate. The text model is trained on synthetic data.
  Do not use as sole basis for determining factual accuracy.
</div>
""", unsafe_allow_html=True)