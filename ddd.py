from neo4j import GraphDatabase

# Update these with your Neo4j instance details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

cypher_commands = """
// ==== PEOPLE ====
CREATE (turing:Person {name:"Alan Turing", desc:"Pioneer of computer science and AI; Turing Test author."});
CREATE (mccarthy:Person {name:"John McCarthy", desc:"Father of AI, LISP inventor, MIT & Stanford AI Labs."});
CREATE (minsky:Person {name:"Marvin Minsky", desc:"Co-founder MIT AI Lab, symbolic AI pioneer."});
CREATE (lecun:Person {name:"Yann LeCun", desc:"CNN pioneer, Meta AI chief scientist, Turing Award winner."});
CREATE (bengio:Person {name:"Yoshua Bengio", desc:"Deep learning leader, MILA founder, Turing Award."});
CREATE (hinton:Person {name:"Geoffrey Hinton", desc:"Godfather of deep learning, U. Toronto, Google Brain."});
CREATE (sutskever:Person {name:"Ilya Sutskever", desc:"Hinton student, AlexNet, OpenAI co-founder."});
CREATE (krizhevsky:Person {name:"Alex Krizhevsky", desc:"AlexNet co-creator, Hinton student."});
CREATE (hassabis:Person {name:"Demis Hassabis", desc:"DeepMind founder/CEO, AlphaGo, neuroscientist."});
CREATE (suleyman:Person {name:"Mustafa Suleyman", desc:"DeepMind/Inflection AI co-founder, AI safety."});
CREATE (legg:Person {name:"Shane Legg", desc:"DeepMind co-founder, AGI risk."});
CREATE (altman:Person {name:"Sam Altman", desc:"OpenAI CEO, ex-Y Combinator president."});
CREATE (musk:Person {name:"Elon Musk", desc:"OpenAI & xAI co-founder, Tesla, SpaceX, etc."});
CREATE (brockman:Person {name:"Greg Brockman", desc:"OpenAI co-founder, ex-Stripe CTO."});
CREATE (zaremba:Person {name:"Wojciech Zaremba", desc:"OpenAI co-founder, LLM/reinforcement learning."});
CREATE (dario:Person {name:"Dario Amodei", desc:"OpenAI VP Research, Anthropic co-founder."});
CREATE (daniela:Person {name:"Daniela Amodei", desc:"Anthropic president, ex-OpenAI."});
CREATE (leike:Person {name:"Jan Leike", desc:"AI alignment specialist, Anthropic, ex-OpenAI."});
CREATE (shazeer:Person {name:"Noam Shazeer", desc:"Transformer co-inventor, Character.AI co-founder."});
CREATE (freitas:Person {name:"Daniel De Freitas", desc:"Character.AI co-founder, ex-Google."});
CREATE (gomez:Person {name:"Aidan Gomez", desc:"Transformer co-inventor, Cohere co-founder."});
CREATE (frosst:Person {name:"Nick Frosst", desc:"Cohere co-founder, ex-Google Brain."});
CREATE (feifei:Person {name:"Fei-Fei Li", desc:"Stanford AI leader, ImageNet, SenseTime advisor."});
CREATE (xuli:Person {name:"Xu Li", desc:"SenseTime co-founder, computer vision."});
CREATE (wangxg:Person {name:"Wang Xiaogang", desc:"SenseTime co-founder."});
CREATE (tang:Person {name:"Tang Xiao'ou", desc:"SenseTime co-founder, Chinese AI professor."});
CREATE (robinli:Person {name:"Robin Li", desc:"Baidu CEO, Chinese search/AI pioneer."});

// ==== COMPANIES/LABS ====
CREATE (mit:Org {name:"MIT AI Lab", desc:"Early AI lab; home of Minsky, McCarthy; robotics, symbolic AI."});
CREATE (stanford:Org {name:"Stanford AI Lab", desc:"Founded by McCarthy; expert systems, robotics."});
CREATE (googlebrain:Org {name:"Google Brain", desc:"Deep learning team at Google, transformer/seq2seq innovators."});
CREATE (deepmind:Org {name:"DeepMind", desc:"London AI startup, AlphaGo, AlphaFold, AGI research."});
CREATE (googledeepmind:Org {name:"Google DeepMind", desc:"Merged DeepMind+Google Brain (2023), unified AI division."});
CREATE (meta:Org {name:"Meta AI (FAIR)", desc:"Meta/Facebook AI, large models (LLaMA), vision."});
CREATE (openai:Org {name:"OpenAI", desc:"San Francisco AI lab; GPT, DALL-E, ChatGPT."});
CREATE (anthropic:Org {name:"Anthropic", desc:"AI safety-focused lab, Claude models, ex-OpenAI founders."});
CREATE (inflection:Org {name:"Inflection AI", desc:"Personal AI assistant Pi, DeepMind alumni founded."});
CREATE (characterai:Org {name:"Character.AI", desc:"AI character chatbots, ex-Google Brain founders."});
CREATE (cohere:Org {name:"Cohere", desc:"Canadian LLM startup, transformer inventors."});
CREATE (ai21:Org {name:"AI21 Labs", desc:"Israeli AI/LLM company."});
CREATE (forum:Org {name:"Frontier Model Forum", desc:"AI safety/standards forum: OpenAI, Google, Anthropic, MS."});
CREATE (baai:Org {name:"Beijing Academy of AI (BAAI)", desc:"China’s top AI research center; WuDao model."});
CREATE (sensetime:Org {name:"SenseTime", desc:"China’s largest AI startup; computer vision, surveillance, AR."});
CREATE (megvii:Org {name:"Megvii", desc:"Chinese facial recognition pioneer, Face++."});
CREATE (yitu:Org {name:"YITU", desc:"Facial recognition, medical AI, security."});
CREATE (iflytek:Org {name:"iFlytek", desc:"China’s voice AI and translation leader."});
CREATE (baidu:Org {name:"Baidu", desc:"China’s Google, leader in AI, NLP."});
CREATE (alibaba:Org {name:"Alibaba DAMO", desc:"Alibaba’s research arm, NLP and vision models."});
CREATE (tencent:Org {name:"Tencent AI Lab", desc:"Tencent’s AI/game/vision research."});
CREATE (zhiguang:Org {name:"Zhiguang AI", desc:"China's leading NLP/LLM startup (GLM models)."});

// ==== MODELS/PRODUCTS ====
CREATE (gpt3:Model {name:"GPT-3", desc:"OpenAI’s large language model, 175B parameters."});
CREATE (gpt4:Model {name:"GPT-4", desc:"OpenAI’s multimodal LLM, powers ChatGPT Plus."});
CREATE (gpt5:Model {name:"GPT-5", desc:"Upcoming frontier LLM by OpenAI."});
CREATE (dalle:Model {name:"DALL-E", desc:"OpenAI’s image generator."});
CREATE (chatgpt:Model {name:"ChatGPT", desc:"OpenAI’s conversational chatbot UI."});
CREATE (claude2:Model {name:"Claude 2", desc:"Anthropic’s safe LLM, ChatGPT rival."});
CREATE (claude3:Model {name:"Claude 3", desc:"Anthropic’s 2024 LLM, strong in reasoning."});
CREATE (alphago:Model {name:"AlphaGo", desc:"DeepMind’s Go champion, beat Lee Sedol."});
CREATE (alphafold:Model {name:"AlphaFold", desc:"DeepMind’s protein folding AI."});
CREATE (llama:Model {name:"LLaMA", desc:"Meta’s open-source LLM."});
CREATE (llama2:Model {name:"Llama 2", desc:"Meta’s 2023 LLM, commercial-friendly."});
CREATE (llama3:Model {name:"Llama 3", desc:"Meta’s 2024 LLM, massive context window."});
CREATE (wudao:Model {name:"WuDao 2.0", desc:"BAAI’s trillion-parameter Chinese/English LLM."});
CREATE (ernie:Model {name:"ERNIE Bot", desc:"Baidu’s GPT-4 rival, Chinese NLP focus."});
CREATE (tongyi:Model {name:"Tongyi Qianwen", desc:"Alibaba’s multi-lingual LLM."});
CREATE (glm130b:Model {name:"GLM-130B", desc:"Zhiguang's large open-source Chinese LLM."});
CREATE (glm4:Model {name:"GLM-4", desc:"Zhiguang's latest Chinese LLM (2024)."});

// ==== RELATIONSHIPS: PEOPLE TO COMPANIES ====
MATCH (turing:Person {name:"Alan Turing"}), (mit:Org {name:"MIT AI Lab"})
CREATE (turing)-[:INSPIRED]->(mit);
MATCH (mccarthy:Person {name:"John McCarthy"}), (mit:Org {name:"MIT AI Lab"})
CREATE (mccarthy)-[:FOUNDED]->(mit);
MATCH (mccarthy:Person {name:"John McCarthy"}), (stanford:Org {name:"Stanford AI Lab"})
CREATE (mccarthy)-[:FOUNDED]->(stanford);
MATCH (minsky:Person {name:"Marvin Minsky"}), (mit:Org {name:"MIT AI Lab"})
CREATE (minsky)-[:COFOUNDED]->(mit);
MATCH (lecun:Person {name:"Yann LeCun"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (lecun)-[:CHIEF_SCIENTIST]->(meta);
MATCH (bengio:Person {name:"Yoshua Bengio"}), (cohere:Org {name:"Cohere"})
CREATE (bengio)-[:ADVISOR]->(cohere);
MATCH (bengio:Person {name:"Yoshua Bengio"}), (ai21:Org {name:"AI21 Labs"})
CREATE (bengio)-[:ADVISOR]->(ai21);
MATCH (hinton:Person {name:"Geoffrey Hinton"}), (googlebrain:Org {name:"Google Brain"})
CREATE (hinton)-[:JOINED]->(googlebrain);
MATCH (sutskever:Person {name:"Ilya Sutskever"}), (openai:Org {name:"OpenAI"})
CREATE (sutskever)-[:COFOUNDED]->(openai);
MATCH (krizhevsky:Person {name:"Alex Krizhevsky"}), (hinton:Person {name:"Geoffrey Hinton"})
CREATE (krizhevsky)-[:STUDENT_OF]->(hinton);
MATCH (hassabis:Person {name:"Demis Hassabis"}), (deepmind:Org {name:"DeepMind"})
CREATE (hassabis)-[:FOUNDED]->(deepmind);
MATCH (suleyman:Person {name:"Mustafa Suleyman"}), (deepmind:Org {name:"DeepMind"})
CREATE (suleyman)-[:COFOUNDED]->(deepmind);
MATCH (legg:Person {name:"Shane Legg"}), (deepmind:Org {name:"DeepMind"})
CREATE (legg)-[:COFOUNDED]->(deepmind);
MATCH (suleyman:Person {name:"Mustafa Suleyman"}), (inflection:Org {name:"Inflection AI"})
CREATE (suleyman)-[:COFOUNDED]->(inflection);
MATCH (altman:Person {name:"Sam Altman"}), (openai:Org {name:"OpenAI"})
CREATE (altman)-[:COFOUNDED]->(openai);
MATCH (musk:Person {name:"Elon Musk"}), (openai:Org {name:"OpenAI"})
CREATE (musk)-[:COFOUNDED]->(openai);
MATCH (brockman:Person {name:"Greg Brockman"}), (openai:Org {name:"OpenAI"})
CREATE (brockman)-[:COFOUNDED]->(openai);
MATCH (zaremba:Person {name:"Wojciech Zaremba"}), (openai:Org {name:"OpenAI"})
CREATE (zaremba)-[:COFOUNDED]->(openai);
MATCH (dario:Person {name:"Dario Amodei"}), (anthropic:Org {name:"Anthropic"})
CREATE (dario)-[:COFOUNDED]->(anthropic);
MATCH (daniela:Person {name:"Daniela Amodei"}), (anthropic:Org {name:"Anthropic"})
CREATE (daniela)-[:COFOUNDED]->(anthropic);
MATCH (leike:Person {name:"Jan Leike"}), (anthropic:Org {name:"Anthropic"})
CREATE (leike)-[:JOINED]->(anthropic);
MATCH (shazeer:Person {name:"Noam Shazeer"}), (characterai:Org {name:"Character.AI"})
CREATE (shazeer)-[:COFOUNDED]->(characterai);
MATCH (freitas:Person {name:"Daniel De Freitas"}), (characterai:Org {name:"Character.AI"})
CREATE (freitas)-[:COFOUNDED]->(characterai);
MATCH (gomez:Person {name:"Aidan Gomez"}), (cohere:Org {name:"Cohere"})
CREATE (gomez)-[:COFOUNDED]->(cohere);
MATCH (frosst:Person {name:"Nick Frosst"}), (cohere:Org {name:"Cohere"})
CREATE (frosst)-[:COFOUNDED]->(cohere);
MATCH (feifei:Person {name:"Fei-Fei Li"}), (sensetime:Org {name:"SenseTime"})
CREATE (feifei)-[:ADVISOR]->(sensetime);
MATCH (xuli:Person {name:"Xu Li"}), (sensetime:Org {name:"SenseTime"})
CREATE (xuli)-[:COFOUNDED]->(sensetime);
MATCH (wangxg:Person {name:"Wang Xiaogang"}), (sensetime:Org {name:"SenseTime"})
CREATE (wangxg)-[:COFOUNDED]->(sensetime);
MATCH (tang:Person {name:"Tang Xiao'ou"}), (sensetime:Org {name:"SenseTime"})
CREATE (tang)-[:COFOUNDED]->(sensetime);
MATCH (robinli:Person {name:"Robin Li"}), (baidu:Org {name:"Baidu"})
CREATE (robinli)-[:FOUNDED]->(baidu);

// ==== COMPANIES TO MODELS ====
MATCH (openai:Org {name:"OpenAI"}), (gpt3:Model {name:"GPT-3"})
CREATE (openai)-[:RELEASED]->(gpt3);
MATCH (openai:Org {name:"OpenAI"}), (gpt4:Model {name:"GPT-4"})
CREATE (openai)-[:RELEASED]->(gpt4);
MATCH (openai:Org {name:"OpenAI"}), (gpt5:Model {name:"GPT-5"})
CREATE (openai)-[:DEVELOPING]->(gpt5);
MATCH (openai:Org {name:"OpenAI"}), (dalle:Model {name:"DALL-E"})
CREATE (openai)-[:RELEASED]->(dalle);
MATCH (openai:Org {name:"OpenAI"}), (chatgpt:Model {name:"ChatGPT"})
CREATE (openai)-[:RELEASED]->(chatgpt);
MATCH (anthropic:Org {name:"Anthropic"}), (claude2:Model {name:"Claude 2"})
CREATE (anthropic)-[:RELEASED]->(claude2);
MATCH (anthropic:Org {name:"Anthropic"}), (claude3:Model {name:"Claude 3"})
CREATE (anthropic)-[:RELEASED]->(claude3);
MATCH (deepmind:Org {name:"DeepMind"}), (alphago:Model {name:"AlphaGo"})
CREATE (deepmind)-[:RELEASED]->(alphago);
MATCH (deepmind:Org {name:"DeepMind"}), (alphafold:Model {name:"AlphaFold"})
CREATE (deepmind)-[:RELEASED]->(alphafold);
MATCH (meta:Org {name:"Meta AI (FAIR)"}), (llama:Model {name:"LLaMA"})
CREATE (meta)-[:RELEASED]->(llama);
MATCH (meta:Org {name:"Meta AI (FAIR)"}), (llama2:Model {name:"Llama 2"})
CREATE (meta)-[:RELEASED]->(llama2);
MATCH (meta:Org {name:"Meta AI (FAIR)"}), (llama3:Model {name:"Llama 3"})
CREATE (meta)-[:RELEASED]->(llama3);
MATCH (baai:Org {name:"Beijing Academy of AI (BAAI)"}), (wudao:Model {name:"WuDao 2.0"})
CREATE (baai)-[:RELEASED]->(wudao);
MATCH (baidu:Org {name:"Baidu"}), (ernie:Model {name:"ERNIE Bot"})
CREATE (baidu)-[:RELEASED]->(ernie);
MATCH (alibaba:Org {name:"Alibaba DAMO"}), (tongyi:Model {name:"Tongyi Qianwen"})
CREATE (alibaba)-[:RELEASED]->(tongyi);
MATCH (zhiguang:Org {name:"Zhiguang AI"}), (glm130b:Model {name:"GLM-130B"})
CREATE (zhiguang)-[:RELEASED]->(glm130b);
MATCH (zhiguang:Org {name:"Zhiguang AI"}), (glm4:Model {name:"GLM-4"})
CREATE (zhiguang)-[:RELEASED]->(glm4);

// ==== COMPANY TO COMPANY CONNECTIONS ====
MATCH (googlebrain:Org {name:"Google Brain"}), (deepmind:Org {name:"DeepMind"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (googlebrain)-[:MERGED_WITH]->(deepmind),
       (deepmind)-[:BECAME]->(googledeepmind);
MATCH (openai:Org {name:"OpenAI"}), (anthropic:Org {name:"Anthropic"})
CREATE (openai)-[:SPAWNED]->(anthropic);
MATCH (forum:Org {name:"Frontier Model Forum"}), (openai:Org {name:"OpenAI"})
CREATE (forum)-[:MEMBER]->(openai);
MATCH (forum:Org {name:"Frontier Model Forum"}), (anthropic:Org {name:"Anthropic"})
CREATE (forum)-[:MEMBER]->(anthropic);
MATCH (forum:Org {name:"Frontier Model Forum"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (forum)-[:MEMBER]->(meta);
MATCH (forum:Org {name:"Frontier Model Forum"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (forum)-[:MEMBER]->(googledeepmind);


"""

def run_cypher_script(uri, user, password, script):
    # Split the script by semicolon (;) to run each command separately
    commands = [cmd.strip() for cmd in script.split(';') if cmd.strip()]
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for cmd in commands:
            try:
                session.run(cmd)
                print(f"Executed: {cmd[:60]}...")
            except Exception as e:
                print(f"Error with command: {cmd[:60]}... \n{e}")
    driver.close()

if __name__ == "__main__":
    run_cypher_script(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, cypher_commands)
    print("All Cypher commands executed.")
