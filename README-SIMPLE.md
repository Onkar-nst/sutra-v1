# Sutra, explained simply

*This is the easy version of the project. You don't need to know any coding to read it.*

---

## The project in one line

**I built, from zero, the kind of tool that is used to make AI. Then I used my tool to make two small AIs: one that reads handwritten numbers and one that writes like Shakespeare.**

---

## First, a few words you should know

| Word | What it means |
|---|---|
| **Code / program** | Instructions written for a computer, step by step, like a recipe. |
| **Python** | A popular language for writing those instructions. |
| **AI (Artificial Intelligence)** | A computer program that can do things we usually think need a brain, like recognising a picture or writing a sentence. |
| **Machine learning** | A way of making AI where we don't write the rules ourselves. We show the computer lots of examples and it **learns** the rules on its own. |
| **Model** | The "brain" the computer builds while learning. |
| **NumPy** | A free Python tool that does maths on big tables of numbers very fast. Think of it as a super calculator. |
| **PyTorch** | A famous ready-made toolkit for building AI, made by Meta (Facebook). Most AI researchers use it. |

---

## How does a computer "learn"?

Imagine a machine with **thousands of knobs** on it. You put in a picture of a handwritten **7**, and the machine gives an answer. At first the knobs are set randomly, so its answer is rubbish, maybe "3".

Now we do three things, again and again:

1. **Guess:** the machine looks at the picture and gives an answer.
2. **Check:** we compare its answer with the right one and give it a **mistake score**. A big score means very wrong.
3. **Adjust:** we turn every knob a tiny bit in the direction that makes the mistake score smaller.

After doing this with thousands of pictures, the knobs end up in positions that give the right answer. **That's all "learning" is: turning knobs until the mistakes become small.**

The hard part is step 3. With thousands of knobs, how do you know **which way to turn each one**? You work **backwards** from the mistake, like a teacher tracing a wrong answer back through each line of a student's working to find where it went wrong. Doing this backward tracing automatically is called **autograd**, and it's the heart of this project.

---

## So what did I actually build?

PyTorch already does all of the above. Normally you just use it, and it hides everything inside.

**I built my own version of PyTorch from scratch and called it Sutra.** I used only NumPy, the calculator, and wrote everything else myself. It's like building your own bicycle from metal pipes instead of buying one from a shop. You end up understanding every nut and bolt.

Sutra has 6 main parts:

| Part | Simple meaning |
|---|---|
| **Tensor** | A grid of numbers. A photo is a grid where each number says how dark one dot (pixel) is. Every tensor remembers how it was made, so the mistake can be traced backwards later. |
| **Autograd** | The "backward tracing" machine. It works out which way to turn every knob. |
| **Layers** | Ready-made building blocks, like LEGO pieces. Some are good at looking at pictures, some at reading text. You stack them to make a model. |
| **Optimizers** | Different rules for **how much** to turn each knob. Some take careful small steps, some speed up when they're going the right way. |
| **Tokenizer** | Breaks text into small pieces (letters or common chunks like "the") and turns them into numbers, because computers only understand numbers. |
| **Quantisation** | Shrinks a finished model to **4 times smaller** by storing each knob with less detail, like writing 3.14 instead of 3.14159265. |

All of this is about **2,200 lines of code**, and none of it is copied from PyTorch or any other AI toolkit.

---

## The two AIs I trained with it

### 1. A model that reads handwritten digits (0 to 9)

- I showed it **55,000 pictures** of handwritten numbers. Each picture is a small 28 × 28 grid of dots.
- It learned for about **3 minutes** on a normal laptop.
- Then I tested it on **10,000 new pictures** it had never seen before.
- It got **99.37% correct**. That means only **63 mistakes out of 10,000**.

### 2. A model that writes like Shakespeare (a mini GPT)

- **GPT** is the same kind of AI that ChatGPT is built on, just much, much smaller.
- The whole idea is simple: **read some text and guess the next letter.** It's like your phone keyboard suggesting the next word, but done one letter at a time.
- I fed it about **1 million letters** of Shakespeare's plays.
- It has about **2.7 million knobs** (ChatGPT-sized models have billions).
- After learning, it writes new text that **looks like** Shakespeare: character names, old English words, the layout of a play. But it doesn't really **mean** anything, because it's tiny.

---

## How do I know my tool is correct?

This is the most important part. A tool that is "almost right" doesn't show an error. It just learns badly, and you never find out why. So I checked everything:

- **Answer key:** for every single calculation, I ran the same numbers through **my tool and through PyTorch**. The answers had to match to about **7 decimal places**. It's like solving a maths problem and checking it against the answer key.
- **207 automatic tests** do this checking again every time I change anything.
- **Testing the tests:** on purpose, I put tiny mistakes into my code, like multiplying by 0.99 instead of 1. The tests caught every one. That proves the tests actually work.

---

## Science experiments I ran

Like a science fair, I changed **one thing at a time**, repeated each test **3 times**, and compared the results fairly:

1. **Does the knob-turning rule (optimizer) matter?** Once each rule was set up properly, all five scored almost the same, about **97.8%**. The setup mattered more than which rule I picked.
2. **Does the starting position of the knobs matter?** Yes, a lot! With a bad start, the model got stuck at about **10%**, which is no better than guessing. With a smart start, it reached **93% to 97%**.
3. **Does a helper layer called "BatchNorm" help?** Without it, the model never got above about **64%**, and at most learning speeds it learned nothing at all. With it, the model scored about **97%** at **every** speed I tried.

I also checked how the model improves as I make it bigger, and which way of splitting text into pieces works best.

---

## Running it inside a web browser

AI usually runs on big computers in the cloud. I wanted anyone to try my models **just by opening a web page**, with nothing to install.

So I wrote a small second engine (only **244 lines**) in **JavaScript**, the language web browsers understand. It gives the same answers as the Python version to about **5 decimal places**.

On the demo page you can:
- ✍️ **Draw a digit** with your mouse and watch the AI guess it.
- 📜 **Watch the mini GPT write Shakespeare** live, letter by letter.
- 📊 **See the charts** from the experiments.

---

## Honest limits

- Sutra is **slower** than PyTorch. PyTorch has had years of work by big teams to make it fast. Sutra was built to be **understood and checked**, not to win races.
- The Shakespeare AI is tiny, so its writing **looks** right but doesn't make real sense.

---

## Why this project matters

Most people who learn AI only **use** tools like PyTorch. This project shows that I understand what happens **inside** them. I built the tool, proved it's correct, used it to train real AI models, ran fair experiments, and put the result on a web page anyone can open.

---

## Want to try it?

If Python is installed on your computer, open a terminal in this folder and type:

```bash
make serve
```

Then open **http://localhost:8000** in your web browser.

*For the full technical version, see [README.md](README.md).*
