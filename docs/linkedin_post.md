# LinkedIn post

Attach: `assets/llm_alignment_flow_vertical.gif` (4:5). Bold text uses Unicode bold characters, so copy-paste it as-is.

Before posting: make the GitHub repo public so the link works.

## Post

🩺 𝗪𝗼𝘂𝗹𝗱 𝘆𝗼𝘂 𝘁𝗿𝘂𝘀𝘁 𝗮 𝗰𝗵𝗮𝘁𝗯𝗼𝘁 𝘄𝗶𝘁𝗵 𝘆𝗼𝘂𝗿 𝗰𝗵𝗲𝘀𝘁 𝗽𝗮𝗶𝗻?

A patient types: "Chest pain spreading to my left arm, what should I do?"

A raw LLM might say: "Rest and see a doctor if it continues." 😬
The right answer: "This could be a heart attack. 𝗖𝗮𝗹𝗹 𝗲𝗺𝗲𝗿𝗴𝗲𝗻𝗰𝘆 𝘀𝗲𝗿𝘃𝗶𝗰𝗲𝘀 𝗻𝗼𝘄." 🚨

In medicine, one bad reply is not a small bug. It can cost a life. So we can't just put an LLM in front of patients and hope for the best.

🔧 𝗛𝗼𝘄 𝘁𝗵𝗲 𝗶𝗻𝗱𝘂𝘀𝘁𝗿𝘆 𝗳𝗶𝘅𝗲𝘀 𝘁𝗵𝗶𝘀: 𝗮𝗹𝗶𝗴𝗻𝗺𝗲𝗻𝘁
1️⃣ 𝗣𝗿𝗲𝘁𝗿𝗮𝗶𝗻𝗶𝗻𝗴: the model reads the internet and learns language
2️⃣ 𝗦𝗙𝗧: it learns to answer like an assistant
3️⃣ 𝗛𝘂𝗺𝗮𝗻 𝗳𝗲𝗲𝗱𝗯𝗮𝗰𝗸: people rank answers, and a reward model learns what "good" means
Then one of three paths:
🎯 𝗣𝗣𝗢: the reward model scores every answer (used for ChatGPT)
👥 𝗚𝗥𝗣𝗢: compare a group of answers and reward the best (DeepSeek-R1)
⚖️ 𝗗𝗣𝗢 / 𝗢𝗥𝗣𝗢: learn directly from good vs bad pairs, no reward model

🧪 𝗪𝗵𝗮𝘁 𝗜 𝗯𝘂𝗶𝗹𝘁: 𝗠𝗲𝗱𝗔𝗹𝗶𝗴𝗻-𝗥𝗟
I took Qwen2.5-0.5B, fine-tuned it on medical Q&A, then aligned it with 𝗣𝗣𝗢 and 𝗚𝗥𝗣𝗢, all on a free Colab T4.

To measure it honestly, I built a 𝟮𝟬𝟬-𝗶𝘁𝗲𝗺 𝗴𝗼𝗹𝗱𝗲𝗻 𝗱𝗮𝘁𝗮𝘀𝗲𝘁:
✅ 150 medical multiple-choice questions
🚨 30 red-flag symptoms the model must escalate
⛔ 20 unsafe requests it must refuse

📊 𝗪𝗵𝗮𝘁 𝗵𝗮𝗽𝗽𝗲𝗻𝗲𝗱
🔸 SFT made the model sound like a doctor but made it 𝗹𝗲𝘀𝘀 𝘀𝗮𝗳𝗲 (overall 0.51 → 0.385)
🔸 My first PPO run 𝗿𝗲𝘄𝗮𝗿𝗱-𝗵𝗮𝗰𝗸𝗲𝗱: 5-word answers and even unsafe drug advice. A stronger KL penalty fixed it.
🔸 𝗚𝗥𝗣𝗢 with simple rule-based rewards recovered the most: 𝟬.𝟯𝟴𝟱 → 𝟬.𝟰𝟰, red-flag detection 0.80 → 0.90
🔸 But the untouched base model still scored best (0.51)

💡 𝗠𝘆 𝘁𝗮𝗸𝗲𝗮𝘄𝗮𝘆: alignment is not a checkbox. Fine-tuning can quietly remove safety, and a reward you don't measure is a reward the model will game. In healthcare, the 𝗲𝘃𝗮𝗹𝘂𝗮𝘁𝗶𝗼𝗻 𝘀𝗲𝘁 𝗺𝗮𝘁𝘁𝗲𝗿𝘀 𝗮𝘀 𝗺𝘂𝗰𝗵 𝗮𝘀 𝘁𝗵𝗲 𝘁𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗺𝗲𝘁𝗵𝗼𝗱.

Next, I want to add rewards that target refusals directly. Unsafe-request refusal never improved, because nothing rewarded it.

👇 Models, dataset and code in the first comment.

#LLM #RLHF #GRPO #PPO #AIinHealthcare #MachineLearning #HuggingFace #GenAI

## First comment

🔗 𝗘𝘃𝗲𝗿𝘆𝘁𝗵𝗶𝗻𝗴 𝗶𝘀 𝗼𝗽𝗲𝗻:

🤗 Models on Hugging Face
• SFT: https://huggingface.co/ubaidii123/medalign-sft
• PPO: https://huggingface.co/ubaidii123/medalign-ppo
• GRPO: https://huggingface.co/ubaidii123/medalign-grpo
📋 Golden eval dataset: https://huggingface.co/datasets/ubaidii123/medalign-golden-eval

💻 Code, notebooks and full results: https://github.com/ubaidur404786/medalign-rl

⚠️ Educational project only, not medical advice.
