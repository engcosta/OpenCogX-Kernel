🧬 Open AGI Kernel — Kernel Specification v0.1
الهدف

تعريف أصغر نواة ممكنة:

قابلة للتنفيذ

قابلة للتوسع

قابلة للفهم

ولا تعتمد على أي نموذج بعينه

الحد الأقصى المتوقع:
👉 500–1000 سطر كود لاحقًا
👉 5 ملفات Core فقط

1️⃣ Kernel Contract (ما الذي تضمنه النواة؟)

النواة ملزمة بأن تكون قادرة على:

تمثيل العالم كحالات وأحداث

تخزين المعرفة والخبرة عبر الزمن

توليد أهداف داخلية

اختيار طريقة التفكير

مراقبة الفشل واقتراح تغييرات

❌ النواة غير مسؤولة عن:

UI

APIs

Tools

Models محددة

Integrations

2️⃣ الملفات الأساسية (غير قابلة للزيادة)
agi_kernel/
├── world.py        # World Model
├── memory.py       # Memory System
├── goals.py        # Goal Engine
├── reasoning.py    # Reasoning Controller
├── meta.py         # Meta-Cognition


أي شيء آخر = Plugin.

3️⃣ world.py — World Model Spec
المسؤولية

تمثيل الواقع كـ حالات → أفعال → نتائج مع عدم يقين.

البيانات الأساسية
State:
  id
  features: dict
  timestamp

Event:
  actor
  action
  context

العلاقات
State --LEADS_TO(p)--> State
Event --CAUSES--> State

API الإلزامي
class WorldModel:
    def observe(event) -> State
    def predict(state, action) -> list[State]
    def confidence(state) -> float

قيود

❌ لا نص خام

❌ لا Embeddings

✅ كل شيء قابل للتفسير

4️⃣ memory.py — Memory System Spec
أنواع الذاكرة
1. Semantic Memory

حقائق

مفاهيم

2. Episodic Memory

(سؤال → إجابة → نتيجة)

3. Temporal Memory

صلاحية المعرفة

التغير عبر الزمن

API الإلزامي
class Memory:
    def store(item, type)
    def recall(query, context)
    def decay()

قوانين

النسيان إجباري

التناقض يُسجَّل لا يُمحى

5️⃣ goals.py — Goal Engine Spec
الهدف

خلق دافع داخلي بدون مستخدم.

أنواع الأهداف
REDUCE_UNCERTAINTY
RESOLVE_CONTRADICTION
IMPROVE_PREDICTION
IMPROVE_SELF_ACCURACY

تمثيل الهدف
Goal:
  type
  priority
  expected_gain

API
class GoalEngine:
    def generate(memory, world) -> list[Goal]
    def prioritize(goals) -> Goal

قيد مهم

❌ لا Goal مصدره المستخدم
(أهداف المستخدم = Inputs، ليست دوافع)

6️⃣ reasoning.py — Reasoning Controller Spec
الدور

التحكم في كيف يفكر النظام وليس بماذا يفكر.

استراتيجيات التفكير (أمثلة)
FAST_RECALL
CAUSAL_REASONING
SIMULATION
VERIFICATION

API
class ReasoningController:
    def choose_strategy(context, hooking_self_model)
    def execute(strategy, context)

قاعدة ذهبية

كل قرار Reasoning يُسجّل سببه.

7️⃣ meta.py — Meta-Cognition Spec
أخطر وأهم ملف
المسؤوليات

تحليل الفشل

اكتشاف أنماط الخطأ

اقتراح تغييرات بنيوية

API
class MetaCognition:
    def evaluate(outcome)
    def detect_pattern(history)
    def propose_change()

أمثلة تغييرات مسموحة

تعديل Ontology

تغيير Strategy default

زيادة وزن نوع ذاكرة

❌ ممنوع:

تعديل الكود ذاتيًا

تغيير قوانين التعلم

8️⃣ القوانين العليا (Kernel Laws)

هذه لا تُمس حتى بالإصدارات القادمة:

Prediction > Memorization

Failure > Success

Contradiction > Confirmation

Strategy Selection > Raw Intelligence

Self-Knowledge > Confidence

9️⃣ Plugin Boundary (خط أحمر)

أي شيء يعتمد على:

LLM

Vector DB

Graph DB

Sensors

Tools

يجب أن يكون:

plugins/
├── llm/
├── vector/
├── graph/
├── tools/


❌ لا Plugin يدخل core/

🔟 Minimal Execution Loop (Skeleton)
while True:
    event = perceive()
    state = world.observe(event)

    goals = goals.generate(memory, world)
    goal = goals.prioritize(goals)

    strategy = reasoning.choose_strategy(state)
    outcome = reasoning.execute(strategy)

    memory.store(outcome)
    meta.evaluate(outcome)


هذا هو الحد الأدنى للحياة.

11️⃣ Definition of “Kernel Complete”

النواة تعتبر مكتملة إذا:

يمكن تشغيلها بدون LLM

يمكن اختبارها بسيناريوهات وهمية

كل قرار قابل للتفسير

كل فشل قابل للتتبع

كل مكوّن قابل للاستبدال