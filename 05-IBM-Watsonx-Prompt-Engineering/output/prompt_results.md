# IBM watsonx.ai Prompt Engineering Results

Model: `ibm/granite-4-h-small`

## Exercise 1: Tokens in a Prompt

Text: `Generative AI models read text as tokens, not words.`

Word count: 9

Token count: 12

Tokens: `['Gener', 'ative', 'ĠAI', 'Ġmodels', 'Ġread', 'Ġtext', 'Ġas', 'Ġtokens', ',', 'Ġnot', 'Ġwords', '.']`

## Exercise 2: Text Generation (Zero-Shot)

No example is given to the model (zero-shot).

**Parameters:** `{'temperature': 0, 'max_tokens': 200}`

**Prompt:**

```
Write a short paragraph of 3 sentences explaining what Generative AI is for a first-year engineering student.
```

**Response:**

```
Generative AI refers to a subset of artificial intelligence that focuses on creating new content, such as images, text, or music, by learning patterns from existing data. It uses techniques like deep learning and neural networks to generate outputs that are often indistinguishable from human-created content. For engineering students, understanding generative AI can open up opportunities in fields like design automation, content creation, and advanced simulations.
```

Total tokens used: 133

## Exercise 3: One-Shot Prompting

One example (product -> slogan) is given before the actual input.

**Parameters:** `{'temperature': 0, 'max_tokens': 30}`

**Prompt:**

```
Convert the product name into a catchy marketing slogan. Reply with only the slogan.

Product: Solar power bank
Slogan: Charge your phone with the power of the sun!

Product: Noise cancelling headphones
Slogan:
```

**Response:**

```
"Immerse in silence, hear the world your way!"
```

Total tokens used: 88

## Exercise 4: Parameter Tuning - Low Temperature (Greedy)

Temperature 0 always picks the most likely next token, so the output is fixed and repeatable.

**Parameters:** `{'temperature': 0, 'max_tokens': 60}`

**Prompt:**

```
Write a two-line poem about the monsoon in Pune.
```

**Response:**

```
Monsoon in Pune, a symphony of rain's sweet song,
Nurturing life, as the city's spirit grows strong.
```

Total tokens used: 70

## Exercise 4b: Parameter Tuning - High Temperature (Sampling)

A high temperature with top-p sampling gives more creative and varied output.

**Parameters:** `{'temperature': 1.2, 'top_p': 0.9, 'max_tokens': 60, 'seed': 42}`

**Prompt:**

```
Write a two-line poem about the monsoon in Pune.
```

**Response:**

```
Raindrops kiss the land of Pune, in monsoon's loving tune,
Washed anew, the city blooms, under the gray, soft afternoon.
```

Total tokens used: 73

## Exercise 5: Parameter Tuning - Max Tokens and Stop Sequence

Max tokens sets the upper limit on output length and the stop sequence '4.' stops the list after 3 points even though 6 were asked for.

**Parameters:** `{'temperature': 0, 'max_tokens': 150, 'stop': ['4.']}`

**Prompt:**

```
List 6 advantages of cloud computing as a numbered list, one line each.
```

**Response:**

```
1. Scalability: Cloud computing allows businesses to easily scale their IT infrastructure up or down based on demand, without the need for significant capital investment.

2. Cost Efficiency: With cloud computing, businesses can reduce their IT costs by eliminating the need for expensive hardware, software, and maintenance.

3. Accessibility: Cloud services can be accessed from anywhere with an internet connection, enabling remote work and collaboration.
```

Total tokens used: 128

## Exercise 6: Summarization

**Parameters:** `{'temperature': 0, 'max_tokens': 200}`

**Prompt:**

```
Summarize the following article in 2 sentences.

Article: India's Chandrayaan-3 mission successfully landed near the south pole of the Moon on 23 August 2023, making India the first country to land in this region. The mission consisted of the Vikram lander and the Pragyan rover. The rover travelled more than 100 metres on the lunar surface and confirmed the presence of sulphur and other elements in the soil. The mission was developed by ISRO at a cost of about 615 crore rupees, which is much lower than similar missions by other countries.
```

**Response:**

```
India's Chandrayaan-3 mission successfully landed near the Moon's south pole on 23 August 2023, making it the first country to achieve this feat, with the mission comprising the Vikram lander and the Pragyan rover. The rover traversed over 100 meters on the lunar surface, confirming the presence of sulphur and other elements in the soil, and the entire mission was developed by ISRO at a cost of approximately 615 crore rupees, significantly lower than similar missions by other countries.
```

Total tokens used: 256

## Exercise 7: Information Extraction

**Parameters:** `{'temperature': 0, 'max_tokens': 150}`

**Prompt:**

```
Extract the following details from the text and return only JSON with keys: mission, organization, landing_date, lander, rover, cost.

Text: India's Chandrayaan-3 mission successfully landed near the south pole of the Moon on 23 August 2023, making India the first country to land in this region. The mission consisted of the Vikram lander and the Pragyan rover. The rover travelled more than 100 metres on the lunar surface and confirmed the presence of sulphur and other elements in the soil. The mission was developed by ISRO at a cost of about 615 crore rupees, which is much lower than similar missions by other countries.
```

**Response:**

```
{
  "mission": "Chandrayaan-3",
  "organization": "ISRO",
  "landing_date": "23 August 2023",
  "lander": "Vikram",
  "rover": "Pragyan",
  "cost": "615 crore rupees"
}
```

Total tokens used: 231

## Exercise 8: Compare Two Texts

**Parameters:** `{'temperature': 0, 'max_tokens': 400}`

**Prompt:**

```
Compare the two laptops below. Give 3 key differences as one-line bullet points and recommend one for a computer engineering student.

Laptop A: Intel i5, 8 GB RAM, 512 GB SSD, integrated graphics, 1.4 kg, battery 12 hours, price 55,000 rupees.
Laptop B: AMD Ryzen 7, 16 GB RAM, 1 TB SSD, RTX 3050 graphics, 2.2 kg, battery 5 hours, price 75,000 rupees.
```

**Response:**

```
Key differences:
- Laptop A has an Intel i5 processor and 8 GB RAM, while Laptop B features an AMD Ryzen 7 processor and 16 GB RAM, offering better multitasking capabilities.
- Laptop A comes with integrated graphics, whereas Laptop B includes an RTX 3050 graphics card, providing superior performance for graphics-intensive tasks.
- Laptop A is lighter at 1.4 kg with a longer battery life of 12 hours, compared to Laptop B's 2.2 kg weight and 5-hour battery life.

Recommendation: For a computer engineering student, I recommend Laptop B due to its superior processing power, increased RAM, dedicated graphics, and larger storage capacity, which are essential for handling complex engineering software and projects.
```

Total tokens used: 291

## Exercise 9: Search

**Parameters:** `{'temperature': 0, 'max_tokens': 60}`

**Prompt:**

```
Here are some documents:
Doc 1: The library is open from 8 AM to 8 PM on weekdays and closed on Sundays.
Doc 2: Students must carry their ID cards to enter the computer lab.
Doc 3: The canteen serves breakfast from 7:30 AM and lunch from 12:30 PM.
Doc 4: Fees for the next semester must be paid before 30 June.
Doc 5: The placement cell conducts mock interviews every Friday.

Which document answers the question: 'When can I practise for job interviews?' Reply with the document number and the answer.
```

**Response:**

```
Doc 5: The placement cell conducts mock interviews every Friday.
```

Total tokens used: 167

## Exercise 10: Classification (Few-Shot)

Few-shot prompting: three labelled examples are given before the new message.

**Parameters:** `{'temperature': 0, 'max_tokens': 5}`

**Prompt:**

```
Classify the customer message as Complaint, Query or Praise. Reply with only the class.

Message: My order arrived broken and nobody is answering my calls.
Class: Complaint

Message: Do you deliver to Nagpur?
Class: Query

Message: The delivery was super fast, thank you!
Class: Praise

Message: I was charged twice for the same item.
Class:
```

**Response:**

```
Complaint
```

Total tokens used: 111

## Exercise 11: RAG (Retrieval Augmented Generation)

Retrieval step: the 2 most relevant sentences are selected from a small knowledge base by keyword overlap. Generation step: the model answers using only that context.

**Parameters:** `{'temperature': 0, 'max_tokens': 80}`

**Prompt:**

```
Answer the question using only the context below. If the answer is not in the context, say 'I don't know'.

Context:
The lab assessment has three components LCA1, LCA2 and LCA3, each of 5 marks.
LCA1 covers assignments 1, 2 and 3. LCA2 covers assignments 4, 5 and 6.

Question: Which assignments are covered in LCA2 and how many marks is it worth?
```

**Response:**

```
LCA2 covers assignments 4, 5, and 6, and it is worth 5 marks.
```

Total tokens used: 151

