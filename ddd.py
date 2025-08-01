// ==== COMPREHENSIVE GLOBAL AI ECOSYSTEM DATABASE ====
// Complete expansion with 150+ companies, 100+ people, 80+ models, complex relationships

// ==== FOUNDATIONAL AI PIONEERS (1940s-1980s) ====
CREATE (turing:Person {name:"Alan Turing", desc:"Pioneer of computer science and AI; Turing Test author.", era:"1940s-1950s", nationality:"British"});
CREATE (mccarthy:Person {name:"John McCarthy", desc:"Father of AI, LISP inventor, MIT & Stanford AI Labs.", era:"1950s-1980s", nationality:"American"});
CREATE (minsky:Person {name:"Marvin Minsky", desc:"Co-founder MIT AI Lab, symbolic AI pioneer.", era:"1950s-2000s", nationality:"American"});
CREATE (newell:Person {name:"Allen Newell", desc:"AI pioneer, Logic Theorist, Carnegie Mellon.", era:"1950s-1990s", nationality:"American"});
CREATE (simon:Person {name:"Herbert Simon", desc:"AI pioneer, Nobel Prize economist, Carnegie Mellon.", era:"1950s-2000s", nationality:"American"});
CREATE (samuel:Person {name:"Arthur Samuel", desc:"Machine learning pioneer, checkers program.", era:"1950s-1980s", nationality:"American"});
CREATE (rosenblatt:Person {name:"Frank Rosenblatt", desc:"Perceptron inventor, neural network pioneer.", era:"1950s-1970s", nationality:"American"});
CREATE (mcculloch:Person {name:"Warren McCulloch", desc:"Neural network theory pioneer.", era:"1940s-1960s", nationality:"American"});
CREATE (pitts:Person {name:"Walter Pitts", desc:"Neural network theory co-pioneer.", era:"1940s-1960s", nationality:"American"});

// ==== MODERN AI LEADERS (1990s-2010s) ====
CREATE (lecun:Person {name:"Yann LeCun", desc:"CNN pioneer, Meta AI chief scientist, Turing Award winner.", era:"1980s-present", nationality:"French"});
CREATE (bengio:Person {name:"Yoshua Bengio", desc:"Deep learning leader, MILA founder, Turing Award.", era:"1990s-present", nationality:"Canadian"});
CREATE (hinton:Person {name:"Geoffrey Hinton", desc:"Godfather of deep learning, U. Toronto, Google Brain.", era:"1980s-present", nationality:"British-Canadian"});
CREATE (schmidhuber:Person {name:"Jürgen Schmidhuber", desc:"LSTM inventor, Swiss AI Institute.", era:"1980s-present", nationality:"German"});
CREATE (hochreiter:Person {name:"Sepp Hochreiter", desc:"LSTM co-inventor, Austrian researcher.", era:"1990s-present", nationality:"Austrian"});
CREATE (vapnik:Person {name:"Vladimir Vapnik", desc:"SVM inventor, statistical learning theory.", era:"1960s-present", nationality:"Russian-American"});
CREATE (pearl:Person {name:"Judea Pearl", desc:"Bayesian networks, causal inference pioneer.", era:"1980s-present", nationality:"Israeli-American"});
CREATE (russell:Person {name:"Stuart Russell", desc:"UC Berkeley, AI textbook author.", era:"1990s-present", nationality:"British-American"});
CREATE (norvig:Person {name:"Peter Norvig", desc:"Ex-Google Research Director, AI textbook author.", era:"1990s-present", nationality:"American"});
CREATE (thrun:Person {name:"Sebastian Thrun", desc:"Stanford AI, Google X, Udacity founder.", era:"2000s-present", nationality:"German-American"});
CREATE (ng:Person {name:"Andrew Ng", desc:"Stanford AI, Coursera, Google Brain, Baidu.", era:"2000s-present", nationality:"British-American"});
CREATE (dean:Person {name:"Jeff Dean", desc:"Google Brain co-founder, TensorFlow architect.", era:"2000s-present", nationality:"American"});

// ==== CURRENT GENERATION AI LEADERS (2010s-present) ====
CREATE (sutskever:Person {name:"Ilya Sutskever", desc:"Hinton student, AlexNet, OpenAI co-founder, SSI.", era:"2010s-present", nationality:"Russian-Canadian"});
CREATE (krizhevsky:Person {name:"Alex Krizhevsky", desc:"AlexNet co-creator, Hinton student.", era:"2010s-present", nationality:"Ukrainian-Canadian"});
CREATE (hassabis:Person {name:"Demis Hassabis", desc:"DeepMind founder/CEO, AlphaGo, neuroscientist.", era:"2010s-present", nationality:"British"});
CREATE (suleyman:Person {name:"Mustafa Suleyman", desc:"DeepMind/Inflection AI co-founder, Microsoft AI.", era:"2010s-present", nationality:"British"});
CREATE (legg:Person {name:"Shane Legg", desc:"DeepMind co-founder, AGI risk researcher.", era:"2010s-present", nationality:"New Zealand"});
CREATE (altman:Person {name:"Sam Altman", desc:"OpenAI CEO, ex-Y Combinator president.", era:"2010s-present", nationality:"American"});
CREATE (musk:Person {name:"Elon Musk", desc:"OpenAI & xAI co-founder, Tesla, SpaceX, etc.", era:"2000s-present", nationality:"South African-American"});
CREATE (brockman:Person {name:"Greg Brockman", desc:"OpenAI co-founder, ex-Stripe CTO.", era:"2010s-present", nationality:"American"});
CREATE (zaremba:Person {name:"Wojciech Zaremba", desc:"OpenAI co-founder, LLM/reinforcement learning.", era:"2010s-present", nationality:"Polish"});
CREATE (schulman:Person {name:"John Schulman", desc:"OpenAI co-founder, RLHF pioneer.", era:"2010s-present", nationality:"American"});
CREATE (chen:Person {name:"Mark Chen", desc:"OpenAI researcher, GPT-4 lead.", era:"2010s-present", nationality:"American"});
CREATE (radford:Person {name:"Alec Radford", desc:"OpenAI researcher, GPT models creator.", era:"2010s-present", nationality:"American"});
CREATE (dario:Person {name:"Dario Amodei", desc:"Anthropic co-founder/CEO, ex-OpenAI VP Research.", era:"2010s-present", nationality:"American"});
CREATE (daniela:Person {name:"Daniela Amodei", desc:"Anthropic president, ex-OpenAI safety.", era:"2010s-present", nationality:"American"});
CREATE (leike:Person {name:"Jan Leike", desc:"AI alignment specialist, Anthropic, ex-OpenAI.", era:"2010s-present", nationality:"German"});
CREATE (christiano:Person {name:"Paul Christiano", desc:"AI alignment researcher, ex-OpenAI.", era:"2010s-present", nationality:"American"});

// ==== TRANSFORMER AND ATTENTION PIONEERS ====
CREATE (vaswani:Person {name:"Ashish Vaswani", desc:"Transformer lead author, ex-Google Brain, Essential AI.", era:"2010s-present", nationality:"Indian-American"});
CREATE (shazeer:Person {name:"Noam Shazeer", desc:"Transformer co-inventor, Character.AI co-founder.", era:"2010s-present", nationality:"American"});
CREATE (parmar:Person {name:"Niki Parmar", desc:"Transformer co-author, ex-Google Brain.", era:"2010s-present", nationality:"Indian-American"});
CREATE (uszkoreit:Person {name:"Jakob Uszkoreit", desc:"Transformer co-author, Inceptive founder.", era:"2010s-present", nationality:"German"});
CREATE (jones:Person {name:"Llion Jones", desc:"Transformer co-author, ex-Google Brain, Sakana AI.", era:"2010s-present", nationality:"Welsh"});
CREATE (gomez:Person {name:"Aidan Gomez", desc:"Transformer co-inventor, Cohere co-founder.", era:"2010s-present", nationality:"Canadian"});
CREATE (kaiser:Person {name:"Łukasz Kaiser", desc:"Transformer co-author, OpenAI researcher.", era:"2010s-present", nationality:"Polish"});
CREATE (polosukhin:Person {name:"Illia Polosukhin", desc:"Transformer co-author, NEAR Protocol founder.", era:"2010s-present", nationality:"Ukrainian"});

// ==== NEW GENERATION STARTUP FOUNDERS (2020s) ====
CREATE (mensch:Person {name:"Arthur Mensch", desc:"Mistral AI co-founder/CEO, ex-DeepMind researcher.", era:"2020s-present", nationality:"French"});
CREATE (lample:Person {name:"Guillaume Lample", desc:"Mistral AI co-founder, ex-Meta FAIR.", era:"2020s-present", nationality:"French"});
CREATE (lacroix:Person {name:"Timothée Lacroix", desc:"Mistral AI co-founder, ex-Meta.", era:"2020s-present", nationality:"French"});
CREATE (holz:Person {name:"David Holz", desc:"Midjourney founder, ex-Leap Motion.", era:"2020s-present", nationality:"American"});
CREATE (mostaque:Person {name:"Emad Mostaque", desc:"Stability AI founder (resigned 2024).", era:"2020s-2024", nationality:"British-Bangladeshi"});
CREATE (brooks:Person {name:"Rodney Brooks", desc:"iRobot co-founder, MIT robotics pioneer.", era:"1980s-present", nationality:"Australian-American"});
CREATE (karpathy:Person {name:"Andrej Karpathy", desc:"Ex-Tesla AI, OpenAI, Stanford researcher.", era:"2010s-present", nationality:"Slovak-Canadian"});
CREATE (freitas:Person {name:"Daniel De Freitas", desc:"Character.AI co-founder, ex-Google.", era:"2010s-present", nationality:"Brazilian-American"});
CREATE (frosst:Person {name:"Nick Frosst", desc:"Cohere co-founder, ex-Google Brain.", era:"2010s-present", nationality:"Canadian"});
CREATE (kon:Person {name:"Ivan Zhang", desc:"Cohere co-founder, ex-Uber.", era:"2010s-present", nationality:"Canadian"});

// ==== JAPANESE AI LEADERS ====
CREATE (ha:Person {name:"David Ha", desc:"Sakana AI co-founder, ex-Google Brain researcher.", era:"2010s-present", nationality:"Canadian-Japanese"});
CREATE (son:Person {name:"Masayoshi Son", desc:"SoftBank founder, AI investor.", era:"1980s-present", nationality:"Japanese"});

// ==== INDIAN AI LEADERS ====
CREATE (srinivasan:Person {name:"Vivek Srinivasan", desc:"Sarvam AI co-founder.", era:"2020s-present", nationality:"Indian"});
CREATE (khandelwal:Person {name:"Pratyush Khandelwal", desc:"Sarvam AI co-founder.", era:"2020s-present", nationality:"Indian"});

// ==== KOREAN AI LEADERS ====
CREATE (kim:Person {name:"Dr. Kim", desc:"LG AI Research leader, ExaONE developer.", era:"2010s-present", nationality:"Korean"});
CREATE (lee:Person {name:"Hae-Jin Lee", desc:"Naver Corporation AI leader.", era:"2010s-present", nationality:"Korean"});

// ==== ROBOTICS AND EMBODIED AI LEADERS ====
CREATE (khatib:Person {name:"Oussama Khatib", desc:"Stanford robotics pioneer.", era:"1980s-present", nationality:"Syrian-American"});
CREATE (raibert:Person {name:"Marc Raibert", desc:"Boston Dynamics founder, legged robotics.", era:"1980s-present", nationality:"American"});
CREATE (siciliano:Person {name:"Bruno Siciliano", desc:"European robotics leader, University of Naples.", era:"1990s-present", nationality:"Italian"});
CREATE (burgard:Person {name:"Wolfram Burgard", desc:"German robotics researcher, Toyota Research.", era:"1990s-present", nationality:"German"});
CREATE (fox:Person {name:"Dieter Fox", desc:"Robotics researcher, University of Washington, NVIDIA.", era:"1990s-present", nationality:"German-American"});

// ==== CHINESE AI ECOSYSTEM LEADERS ====
CREATE (feifei:Person {name:"Fei-Fei Li", desc:"Stanford AI leader, ImageNet, Human-Centered AI Institute.", era:"2000s-present", nationality:"Chinese-American"});
CREATE (xuli:Person {name:"Xu Li", desc:"SenseTime co-founder, computer vision.", era:"2010s-present", nationality:"Chinese"});
CREATE (wangxg:Person {name:"Wang Xiaogang", desc:"SenseTime co-founder, CUHK professor.", era:"2010s-present", nationality:"Chinese"});
CREATE (tang:Person {name:"Tang Xiao'ou", desc:"SenseTime co-founder, Chinese AI professor.", era:"2000s-present", nationality:"Chinese"});
CREATE (robinli:Person {name:"Robin Li", desc:"Baidu CEO, Chinese search/AI pioneer.", era:"2000s-present", nationality:"Chinese"});
CREATE (wanghaifeng:Person {name:"Wang Haifeng", desc:"Baidu CTO, NLP researcher.", era:"2010s-present", nationality:"Chinese"});
CREATE (zhengkai:Person {name:"Kai-Fu Lee", desc:"Sinovation Ventures, ex-Google China, Microsoft.", era:"1990s-present", nationality:"Taiwanese-American"});
CREATE (liangw:Person {name:"Liang Wenfeng", desc:"DeepSeek founder, hedge fund manager.", era:"2020s-present", nationality:"Chinese"});
CREATE (yangqing:Person {name:"Yang Qing", desc:"Alibaba DAMO, ex-Facebook, Caffe creator.", era:"2010s-present", nationality:"Chinese-American"});

// ==== ACADEMIC INSTITUTIONS (Global) ====
CREATE (mit:Org {name:"MIT AI Lab", desc:"Early AI lab; home of Minsky, McCarthy; robotics, symbolic AI.", type:"Academic", location:"Cambridge, MA"});
CREATE (stanford:Org {name:"Stanford AI Lab", desc:"Founded by McCarthy; expert systems, robotics, HAI.", type:"Academic", location:"Stanford, CA"});
CREATE (cmu:Org {name:"Carnegie Mellon AI", desc:"Robotics, machine learning, autonomous vehicles.", type:"Academic", location:"Pittsburgh, PA"});
CREATE (berkeley:Org {name:"UC Berkeley AI", desc:"BAIR lab, robotics, computer vision.", type:"Academic", location:"Berkeley, CA"});
CREATE (toronto:Org {name:"University of Toronto", desc:"Hinton's home, Vector Institute.", type:"Academic", location:"Toronto, Canada"});
CREATE (montreal:Org {name:"MILA Montreal", desc:"Bengio's lab, deep learning research.", type:"Academic", location:"Montreal, Canada"});
CREATE (oxford:Org {name:"Oxford AI", desc:"DeepMind partnership, Future of Humanity Institute.", type:"Academic", location:"Oxford, UK"});
CREATE (cambridge:Org {name:"Cambridge AI", desc:"Machine learning, computational biology.", type:"Academic", location:"Cambridge, UK"});
CREATE (imperial:Org {name:"Imperial College AI", desc:"London AI research, robotics.", type:"Academic", location:"London, UK"});
CREATE (ethz:Org {name:"ETH Zurich AI", desc:"Swiss AI research, Schmidhuber connection.", type:"Academic", location:"Zurich, Switzerland"});
CREATE (epfl:Org {name:"EPFL AI", desc:"Swiss AI research, brain simulation.", type:"Academic", location:"Lausanne, Switzerland"});
CREATE (technion:Org {name:"Technion AI", desc:"Israeli Institute of Technology AI research.", type:"Academic", location:"Haifa, Israel"});
CREATE (tsinghua:Org {name:"Tsinghua University", desc:"Top Chinese university, AI research.", type:"Academic", location:"Beijing, China"});
CREATE (peking:Org {name:"Peking University", desc:"Premier Chinese university, AI lab.", type:"Academic", location:"Beijing, China"});
CREATE (nus:Org {name:"National University of Singapore", desc:"Leading Asian AI research.", type:"Academic", location:"Singapore"});
CREATE (tokyo:Org {name:"University of Tokyo", desc:"Japan's premier AI research institution.", type:"Academic", location:"Tokyo, Japan"});
CREATE (kaist:Org {name:"KAIST", desc:"Korean Advanced Institute of Science and Technology.", type:"Academic", location:"Daejeon, South Korea"});

// ==== MAJOR TECH COMPANIES (Expanded) ====
CREATE (googlebrain:Org {name:"Google Brain", desc:"Deep learning team at Google, transformer/seq2seq innovators.", type:"Big Tech", location:"Mountain View, CA"});
CREATE (deepmind:Org {name:"DeepMind", desc:"London AI startup, AlphaGo, AlphaFold, AGI research.", type:"Big Tech", location:"London, UK"});
CREATE (googledeepmind:Org {name:"Google DeepMind", desc:"Merged DeepMind+Google Brain (2023), unified AI division.", type:"Big Tech", location:"London, UK"});
CREATE (meta:Org {name:"Meta AI (FAIR)", desc:"Meta/Facebook AI, large models (LLaMA), vision.", type:"Big Tech", location:"Menlo Park, CA"});
CREATE (microsoft:Org {name:"Microsoft Research", desc:"AI research, Azure AI, OpenAI partnership.", type:"Big Tech", location:"Redmond, WA"});
CREATE (microsoft_copilot:Org {name:"Microsoft Copilot", desc:"AI assistant platform powered by GPT-4, integrated across Microsoft products.", type:"Big Tech", location:"Redmond, WA"});
CREATE (apple:Org {name:"Apple AI/ML", desc:"On-device AI, Siri, privacy-focused ML.", type:"Big Tech", location:"Cupertino, CA"});
CREATE (amazon:Org {name:"Amazon AI", desc:"Alexa, AWS AI services, robotics.", type:"Big Tech", location:"Seattle, WA"});
CREATE (tesla:Org {name:"Tesla AI", desc:"Autopilot, Full Self-Driving, humanoid robots.", type:"Big Tech", location:"Austin, TX"});
CREATE (netflix:Org {name:"Netflix ML", desc:"Recommendation systems, content optimization.", type:"Big Tech", location:"Los Gatos, CA"});
CREATE (uber:Org {name:"Uber AI", desc:"Autonomous vehicles, optimization, ML platform.", type:"Big Tech", location:"San Francisco, CA"});
CREATE (salesforce:Org {name:"Salesforce AI", desc:"Einstein AI, CRM intelligence.", type:"Big Tech", location:"San Francisco, CA"});
CREATE (adobe:Org {name:"Adobe AI", desc:"Creative AI, Sensei platform.", type:"Big Tech", location:"San Jose, CA"});

// ==== HARDWARE AND INFRASTRUCTURE ====
CREATE (nvidia:Org {name:"NVIDIA AI", desc:"GPU computing, AI chips, software stack.", type:"Hardware", location:"Santa Clara, CA"});
CREATE (intel:Org {name:"Intel AI", desc:"AI chips, Habana Labs, edge computing.", type:"Hardware", location:"Santa Clara, CA"});
CREATE (amd:Org {name:"AMD AI", desc:"GPU computing, data center accelerators.", type:"Hardware", location:"Santa Clara, CA"});
CREATE (qualcomm:Org {name:"Qualcomm AI", desc:"Mobile AI chips, edge computing.", type:"Hardware", location:"San Diego, CA"});
CREATE (groq:Org {name:"Groq", desc:"AI chip startup building LPU ASIC for LLM inference.", type:"Hardware", location:"Mountain View, CA"});
CREATE (cerebras:Org {name:"Cerebras", desc:"Wafer-scale AI chips for training.", type:"Hardware", location:"Sunnyvale, CA"});
CREATE (graphcore:Org {name:"Graphcore", desc:"IPU chips for AI workloads (acquired by SoftBank).", type:"Hardware", location:"Bristol, UK"});
CREATE (sambanova:Org {name:"SambaNova", desc:"AI chip and platform company.", type:"Hardware", location:"Palo Alto, CA"});
CREATE (habana:Org {name:"Habana Labs", desc:"AI training/inference chips (acquired by Intel).", type:"Hardware", location:"Tel Aviv, Israel"});
CREATE (cambricon:Org {name:"Cambricon", desc:"Chinese AI chip unicorn.", type:"Hardware", location:"Beijing, China"});
CREATE (bitmain:Org {name:"Bitmain", desc:"Chinese chip company, AI accelerators.", type:"Hardware", location:"Beijing, China"});

// ==== AI STARTUPS (Tier 1 - Major) ====
CREATE (openai:Org {name:"OpenAI", desc:"San Francisco AI lab; GPT, DALL-E, ChatGPT.", type:"Startup", location:"San Francisco, CA", valuation:"$157B"});
CREATE (anthropic:Org {name:"Anthropic", desc:"AI safety-focused lab, Claude models, ex-OpenAI founders.", type:"Startup", location:"San Francisco, CA", valuation:"$18.4B"});
CREATE (inflection:Org {name:"Inflection AI", desc:"Personal AI assistant Pi, DeepMind alumni founded.", type:"Startup", location:"Palo Alto, CA", valuation:"$4B"});
CREATE (characterai:Org {name:"Character.AI", desc:"AI character chatbots, ex-Google Brain founders.", type:"Startup", location:"Menlo Park, CA", valuation:"$1B+"});
CREATE (cohere:Org {name:"Cohere", desc:"Canadian LLM startup, transformer inventors, enterprise focus.", type:"Startup", location:"Toronto, Canada", valuation:"$5.5B"});
CREATE (ai21:Org {name:"AI21 Labs", desc:"Israeli AI/LLM company, Jurassic models.", type:"Startup", location:"Tel Aviv, Israel", valuation:"$1.4B"});
CREATE (mistral:Org {name:"Mistral AI", desc:"French LLM startup, open-source models, European champion.", type:"Startup", location:"Paris, France", valuation:"$6B"});
CREATE (xai:Org {name:"xAI", desc:"Elon Musk's AI company (founded 2023), develops Grok models.", type:"Startup", location:"San Francisco, CA", valuation:"$50B"});
CREATE (perplexity:Org {name:"Perplexity AI", desc:"AI-powered search engine with reasoning capabilities.", type:"Startup", location:"San Francisco, CA", valuation:"$9B"});
CREATE (aleph_alpha:Org {name:"Aleph Alpha", desc:"German LLM company focusing on European data sovereignty.", type:"Startup", location:"Heidelberg, Germany", valuation:"$500M"});

// ==== AI STARTUPS (Tier 2 - Creative/Specialized) ====
CREATE (midjourney:Org {name:"Midjourney", desc:"AI image generation, Discord-based interface.", type:"Startup", location:"San Francisco, CA"});
CREATE (stability:Org {name:"Stability AI", desc:"Stable Diffusion creators, open-source AI.", type:"Startup", location:"London, UK"});
CREATE (runwayml:Org {name:"Runway ML", desc:"Creative AI tools, video generation.", type:"Startup", location:"New York, NY"});
CREATE (synthesia:Org {name:"Synthesia", desc:"AI video generation with avatars.", type:"Startup", location:"London, UK"});
CREATE (jasper:Org {name:"Jasper AI", desc:"AI writing assistant for marketing.", type:"Startup", location:"Austin, TX"});
CREATE (copy:Org {name:"Copy.ai", desc:"AI copywriting and content generation.", type:"Startup", location:"Memphis, TN"});
CREATE (grammarly:Org {name:"Grammarly", desc:"AI writing assistant, grammar checking.", type:"Startup", location:"San Francisco, CA"});
CREATE (notion:Org {name:"Notion AI", desc:"AI-powered productivity and note-taking.", type:"Startup", location:"San Francisco, CA"});
CREATE (you:Org {name:"You.com", desc:"AI search engine with chatbot integration.", type:"Startup", location:"Palo Alto, CA"});
CREATE (chai:Org {name:"Chai AI", desc:"Consumer AI chatbot platform for entertainment.", type:"Startup", location:"Tel Aviv, Israel"});
CREATE (huggingface:Org {name:"Hugging Face", desc:"Open-source AI platform, model hub, European startup.", type:"Startup", location:"Paris, France"});

// ==== AI STARTUPS (Tier 3 - Enterprise/Vertical) ====
CREATE (adept:Org {name:"Adept AI", desc:"AI assistant for software workflows.", type:"Startup", location:"San Francisco, CA"});
CREATE (harvey:Org {name:"Harvey AI", desc:"AI for legal professionals.", type:"Startup", location:"San Francisco, CA"});
CREATE (typeface:Org {name:"Typeface", desc:"Enterprise generative AI platform.", type:"Startup", location:"San Francisco, CA"});
CREATE (replica:Org {name:"Replika", desc:"AI companion chatbots.", type:"Startup", location:"San Francisco, CA"});
CREATE (avanade:Org {name:"Avanade AI", desc:"Microsoft partner, enterprise AI consulting.", type:"Startup", location:"Seattle, WA"});
CREATE (scale:Org {name:"Scale AI", desc:"AI data platform, training data services.", type:"Startup", location:"San Francisco, CA"});
CREATE (dataminr:Org {name:"Dataminr", desc:"Real-time AI for risk detection.", type:"Startup", location:"New York, NY"});
CREATE (palantir:Org {name:"Palantir", desc:"Big data analytics, AI-powered insights.", type:"Startup", location:"Denver, CO"});
CREATE (c3ai:Org {name:"C3.ai", desc:"Enterprise AI applications platform.", type:"Startup", location:"Redwood City, CA"});
CREATE (databricks:Org {name:"Databricks", desc:"Unified data and AI platform.", type:"Startup", location:"San Francisco, CA"});

// ==== ROBOTICS COMPANIES ====
CREATE (boston_dynamics:Org {name:"Boston Dynamics", desc:"Advanced robotics, Atlas humanoid robot.", type:"Robotics", location:"Waltham, MA"});
CREATE (agility:Org {name:"Agility Robotics", desc:"Bipedal robots for logistics.", type:"Robotics", location:"Corvallis, OR"});
CREATE (figure:Org {name:"Figure AI", desc:"Humanoid robots for workforce automation.", type:"Robotics", location:"Sunnyvale, CA"});
CREATE (1x:Org {name:"1X Technologies", desc:"Humanoid robots, ex-Halodi Robotics.", type:"Robotics", location:"Moss, Norway"});
CREATE (sanctuary:Org {name:"Sanctuary AI", desc:"General-purpose humanoid robots.", type:"Robotics", location:"Vancouver, Canada"});
CREATE (toyota_research:Org {name:"Toyota Research Institute", desc:"Autonomous vehicles and home robots.", type:"Robotics", location:"Los Altos, CA"});
CREATE (waymo:Org {name:"Waymo", desc:"Google's autonomous vehicle division.", type:"Robotics", location:"Mountain View, CA"});
CREATE (cruise:Org {name:"Cruise", desc:"GM's autonomous vehicle startup.", type:"Robotics", location:"San Francisco, CA"});

// ==== CHINESE AI ECOSYSTEM (Expanded) ====
CREATE (baai:Org {name:"Beijing Academy of AI (BAAI)", desc:"China's top AI research center; WuDao model.", type:"Research", location:"Beijing, China"});
CREATE (sensetime:Org {name:"SenseTime", desc:"China's largest AI startup; computer vision, surveillance, AR.", type:"Startup", location:"Hong Kong, China"});
CREATE (megvii:Org {name:"Megvii", desc:"Chinese facial recognition pioneer, Face++.", type:"Startup", location:"Beijing, China"});
CREATE (yitu:Org {name:"YITU", desc:"Facial recognition, medical AI, security.", type:"Startup", location:"Shanghai, China"});
CREATE (iflytek:Org {name:"iFlytek", desc:"China's voice AI and translation leader.", type:"Startup", location:"Hefei, China"});
CREATE (baidu:Org {name:"Baidu", desc:"China's Google, leader in AI, NLP, autonomous driving.", type:"Big Tech", location:"Beijing, China"});
CREATE (alibaba:Org {name:"Alibaba DAMO", desc:"Alibaba's research arm, NLP and vision models.", type:"Big Tech", location:"Hangzhou, China"});
CREATE (alibaba_cloud:Org {name:"Alibaba Cloud", desc:"Cloud computing platform with integrated AI services.", type:"Big Tech", location:"Hangzhou, China"});
CREATE (tencent:Org {name:"Tencent AI Lab", desc:"Tencent's AI/game/vision research.", type:"Big Tech", location:"Shenzhen, China"});
CREATE (zhipu:Org {name:"Zhipu AI", desc:"Chinese LLM startup, GLM models, ChatGLM.", type:"Startup", location:"Beijing, China"});
CREATE (deepseek:Org {name:"DeepSeek", desc:"Chinese AI startup from Hangzhou; cost-efficient open LLMs.", type:"Startup", location:"Hangzhou, China"});
CREATE (minimax:Org {name:"MiniMax", desc:"Chinese AI startup, multimodal models, text-to-video.", type:"Startup", location:"Beijing, China"});
CREATE (moonshot:Org {name:"Moonshot AI", desc:"Chinese LLM startup, Kimi Chat.", type:"Startup", location:"Beijing, China"});
CREATE (stepfun:Org {name:"StepFun", desc:"Chinese AI startup, Step series models.", type:"Startup", location:"Beijing, China"});
CREATE (01ai:Org {name:"01.AI", desc:"Kai-Fu Lee's LLM startup, Yi models.", type:"Startup", location:"Beijing, China"});
CREATE (zhenshen:Org {name:"Zhenshen", desc:"Chinese AI startup focused on reasoning.", type:"Startup", location:"Shanghai, China"});

// ==== INDIAN AI ECOSYSTEM ====
CREATE (sarvam:Org {name:"Sarvam AI", desc:"Multilingual Indian LLM startup, IndiaAI Mission participant.", type:"Startup", location:"Bangalore, India"});
CREATE (corover:Org {name:"CoRover.ai", desc:"Indian conversational AI platform, IndiaAI Mission.", type:"Startup", location:"Mumbai, India"});
CREATE (ganai:Org {name:"Gan AI Labs", desc:"Indian AI research lab, foundational models.", type:"Startup", location:"Bangalore, India"});
CREATE (gnani:Org {name:"Gnani.ai", desc:"Indian speech AI company, IndiaAI Mission.", type:"Startup", location:"Bangalore, India"});
CREATE (soket:Org {name:"Soket AI Labs", desc:"Indian AI startup, sovereign models.", type:"Startup", location:"Delhi, India"});
CREATE (tech_mahindra:Org {name:"Tech Mahindra", desc:"Indian IT services, Project Indus LLM.", type:"Big Tech", location:"Pune, India"});
CREATE (indiaai:Org {name:"IndiaAI Mission", desc:"Government initiative for sovereign AI models.", type:"Government", location:"New Delhi, India"});

// ==== JAPANESE AI ECOSYSTEM ====
CREATE (sakana:Org {name:"Sakana AI", desc:"Japanese AI startup focused on collective intelligence, founded by David Ha and Llion Jones.", type:"Startup", location:"Tokyo, Japan"});
CREATE (softbank:Org {name:"SoftBank", desc:"Japanese conglomerate, major AI investor.", type:"Big Tech", location:"Tokyo, Japan"});
CREATE (sb_openai:Org {name:"SB OpenAI Japan", desc:"Joint venture between SoftBank and OpenAI for Japanese market.", type:"Joint Venture", location:"Tokyo, Japan"});
CREATE (preferred_networks:Org {name:"Preferred Networks", desc:"Japanese AI/robotics company, Toyota partnership.", type:"Startup", location:"Tokyo, Japan"});
CREATE (hitachi:Org {name:"Hitachi AI", desc:"Japanese industrial conglomerate AI division.", type:"Big Tech", location:"Tokyo, Japan"});

// ==== KOREAN AI ECOSYSTEM ====
CREATE (lg_ai:Org {name:"LG AI Research", desc:"LG's AI research division, ExaONE developer.", type:"Big Tech", location:"Seoul, South Korea"});
CREATE (naver:Org {name:"Naver Corporation", desc:"Korean tech giant, HyperClova X developer.", type:"Big Tech", location:"Seongnam, South Korea"});
CREATE (kt:Org {name:"KT Corporation", desc:"Korean telecom developing Korean-language LLMs.", type:"Big Tech", location:"Seoul, South Korea"});
CREATE (upstage:Org {name:"Upstage", desc:"Korean AI startup focusing on enterprise solutions.", type:"Startup", location:"Seoul, South Korea"});
CREATE (hansam_global:Org {name:"Hansam Global", desc:"Korean AI localization company.", type:"Startup", location:"Seoul, South Korea"});
CREATE (hansam:Org {name:"Hansam", desc:"Korean AI company for language processing.", type:"Startup", location:"Seoul, South Korea"});
CREATE (joel:Org {name:"Joel Localization", desc:"Korean AI-powered localization services.", type:"Startup", location:"Seoul, South Korea"});

// ==== EUROPEAN AI ECOSYSTEM ====
CREATE (deepl:Org {name:"DeepL", desc:"German AI translation company.", type:"Startup", location:"Cologne, Germany"});
CREATE (poolside:Org {name:"Poolside", desc:"AI coding assistant, European startup.", type:"Startup", location:"Paris, France"});
CREATE (lightspeed:Org {name:"LightSpeed Studios", desc:"Gaming AI, Tencent subsidiary.", type:"Startup", location:"Montreal, Canada"});

// ==== INVESTMENT AND VC FIRMS ====
CREATE (a16z:Org {name:"Andreessen Horowitz", desc:"Leading VC firm investing in AI startups.", type:"Investment", location:"Menlo Park, CA"});
CREATE (sequoia:Org {name:"Sequoia Capital", desc:"Premier VC firm, OpenAI investor.", type:"Investment", location:"Menlo Park, CA"});
CREATE (khosla:Org {name:"Khosla Ventures", desc:"Vinod Khosla's VC firm, AI focus.", type:"Investment", location:"Menlo Park, CA"});
CREATE (greylock:Org {name:"Greylock Partners", desc:"VC firm investing in AI companies.", type:"Investment", location:"Menlo Park, CA"});
CREATE (gcv:Org {name:"Google Ventures", desc:"Google's VC arm, AI investments.", type:"Investment", location:"Mountain View, CA"});
CREATE (microsoft_ventures:Org {name:"Microsoft Ventures", desc:"Microsoft's investment arm.", type:"Investment", location:"Redmond, WA"});
CREATE (sinovation:Org {name:"Sinovation Ventures", desc:"Kai-Fu Lee's China-focused VC firm.", type:"Investment", location:"Beijing, China"});

// ==== GOVERNMENT AND POLICY ORGANIZATIONS ====
CREATE (nist:Org {name:"NIST AI Risk Management", desc:"US government AI standards and risk assessment.", type:"Government", location:"Gaithersburg, MD"});
CREATE (ostp:Org {name:"OSTP", desc:"White House Office of Science and Technology Policy.", type:"Government", location:"Washington, DC"});
CREATE (darpa:Org {name:"DARPA", desc:"US Defense Advanced Research Projects Agency.", type:"Government", location:"Arlington, VA"});
CREATE (nsf:Org {name:"NSF", desc:"National Science Foundation, AI research funding.", type:"Government", location:"Alexandria, VA"});
CREATE (nih:Org {name:"NIH", desc:"National Institutes of Health, medical AI research.", type:"Government", location:"Bethesda, MD"});
CREATE (eu_ai_office:Org {name:"EU AI Office", desc:"European Union AI regulation and policy.", type:"Government", location:"Brussels, Belgium"});
CREATE (uk_ai_safety:Org {name:"UK AI Safety Institute", desc:"British AI safety research institute.", type:"Government", location:"London, UK"});

// ==== AI MODELS AND PRODUCTS (Comprehensive) ====
// OpenAI Models
CREATE (gpt1:Model {name:"GPT-1", desc:"OpenAI's first GPT model (2018), 117M parameters.", generation:"GPT-1", year:2018, parameters:"117M"});
CREATE (gpt2:Model {name:"GPT-2", desc:"OpenAI's second GPT model (2019), 1.5B parameters.", generation:"GPT-2", year:2019, parameters:"1.5B"});
CREATE (gpt3:Model {name:"GPT-3", desc:"OpenAI's breakthrough LLM, 175B parameters.", generation:"GPT-3", year:2020, parameters:"175B"});
CREATE (gpt35:Model {name:"GPT-3.5", desc:"OpenAI's improved GPT-3, ChatGPT base.", generation:"GPT-3.5", year:2022, parameters:"175B"});
CREATE (gpt4:Model {name:"GPT-4", desc:"OpenAI's multimodal LLM, reasoning breakthrough.", generation:"GPT-4", year:2023, parameters:"1.8T"});
CREATE (gpt4o:Model {name:"GPT-4o", desc:"OpenAI's omni-modal model (2024).", generation:"GPT-4o", year:2024, parameters:"Unknown"});
CREATE (gpt5:Model {name:"GPT-5", desc:"OpenAI's next-generation model (development).", generation:"GPT-5", year:2025, parameters:"Unknown"});
CREATE (o1:Model {name:"o1", desc:"OpenAI's reasoning model series (2024).", generation:"o1", year:2024, parameters:"Unknown"});
CREATE (o3:Model {name:"o3", desc:"OpenAI's latest reasoning model (2024).", generation:"o3", year:2024, parameters:"Unknown"});
CREATE (dalle:Model {name:"DALL-E", desc:"OpenAI's first image generator.", type:"Image", year:2021});
CREATE (dalle2:Model {name:"DALL-E 2", desc:"OpenAI's improved image generator.", type:"Image", year:2022});
CREATE (dalle3:Model {name:"DALL-E 3", desc:"OpenAI's latest image generator.", type:"Image", year:2023});
CREATE (sora:Model {name:"Sora", desc:"OpenAI's video generation model.", type:"Video", year:2024});
CREATE (chatgpt:Model {name:"ChatGPT", desc:"OpenAI's conversational interface.", type:"Interface", year:2022});

// Microsoft Models
CREATE (prometheus:Model {name:"Prometheus", desc:"Microsoft's internal LLM for Copilot enhancements.", generation:"Microsoft", year:2024});
CREATE (copilot_gpt4:Model {name:"Copilot GPT-4", desc:"Microsoft's customized GPT-4 for Copilot integration.", generation:"Copilot", year:2023});

// Anthropic Models
CREATE (claude1:Model {name:"Claude 1", desc:"Anthropic's first constitutional AI model.", generation:"Claude-1", year:2022});
CREATE (claude2:Model {name:"Claude 2", desc:"Anthropic's improved safety-focused LLM.", generation:"Claude-2", year:2023});
CREATE (claude3_haiku:Model {name:"Claude 3 Haiku", desc:"Anthropic's fast, lightweight model.", generation:"Claude-3", year:2024});
CREATE (claude3_sonnet:Model {name:"Claude 3 Sonnet", desc:"Anthropic's balanced performance model.", generation:"Claude-3", year:2024});
CREATE (claude3_opus:Model {name:"Claude 3 Opus", desc:"Anthropic's most capable model.", generation:"Claude-3", year:2024});
CREATE (claude35_sonnet:Model {name:"Claude 3.5 Sonnet", desc:"Anthropic's enhanced Sonnet model.", generation:"Claude-3.5", year:2024});

// Google/DeepMind Models
CREATE (lamda:Model {name:"LaMDA", desc:"Google's conversational AI model.", generation:"LaMDA", year:2021});
CREATE (palm:Model {name:"PaLM", desc:"Google's Pathways Language Model.", generation:"PaLM", year:2022, parameters:"540B"});
CREATE (palm2:Model {name:"PaLM 2", desc:"Google's improved language model.", generation:"PaLM-2", year:2023});
CREATE (gemini_nano:Model {name:"Gemini Nano", desc:"Google's lightweight multimodal model.", generation:"Gemini", year:2023});
CREATE (gemini_pro:Model {name:"Gemini Pro", desc:"Google's mid-tier multimodal model.", generation:"Gemini", year:2023});
CREATE (gemini_ultra:Model {name:"Gemini Ultra", desc:"Google's most capable multimodal model.", generation:"Gemini", year:2023});
CREATE (gemini_15_pro:Model {name:"Gemini 1.5 Pro", desc:"Google's enhanced model with 1M+ context.", generation:"Gemini-1.5", year:2024});
CREATE (bard:Model {name:"Bard", desc:"Google's ChatGPT competitor interface.", type:"Interface", year:2023});
CREATE (alphago:Model {name:"AlphaGo", desc:"DeepMind's Go champion AI.", type:"Game AI", year:2016});
CREATE (alphazero:Model {name:"AlphaZero", desc:"DeepMind's general game AI.", type:"Game AI", year:2017});
CREATE (alphafold:Model {name:"AlphaFold", desc:"DeepMind's protein folding breakthrough.", type:"Science", year:2020});
CREATE (alphafold3:Model {name:"AlphaFold 3", desc:"DeepMind's enhanced protein structure prediction.", type:"Science", year:2024});

// Meta Models
CREATE (llama:Model {name:"LLaMA", desc:"Meta's foundation language model.", generation:"LLaMA-1", year:2023, parameters:"65B"});
CREATE (llama2:Model {name:"Llama 2", desc:"Meta's open commercial model.", generation:"LLaMA-2", year:2023, parameters:"70B"});
CREATE (llama3:Model {name:"Llama 3", desc:"Meta's latest open model family.", generation:"LLaMA-3", year:2024, parameters:"405B"});
CREATE (codellama:Model {name:"Code Llama", desc:"Meta's code generation model.", generation:"Code Llama", year:2023});

// Cohere Models
CREATE (command:Model {name:"Command", desc:"Cohere's enterprise-focused LLM.", generation:"Command", year:2022});
CREATE (command_r:Model {name:"Command R", desc:"Cohere's retrieval-augmented model.", generation:"Command R", year:2024});
CREATE (aya_vision:Model {name:"Aya Vision", desc:"Cohere's multimodal model with vision capabilities.", generation:"Aya", year:2024});

// Mistral Models
CREATE (mistral7b:Model {name:"Mistral 7B", desc:"Mistral's open-source 7B parameter model.", generation:"Mistral", year:2023, parameters:"7B"});
CREATE (mixtral:Model {name:"Mixtral 8x7B", desc:"Mistral's mixture-of-experts model.", generation:"Mixtral", year:2023, parameters:"46.7B"});
CREATE (mistral_large:Model {name:"Mistral Large", desc:"Mistral's flagship commercial model.", generation:"Mistral Large", year:2024});
CREATE (codestral:Model {name:"Codestral", desc:"Mistral's specialized code generation model.", generation:"Codestral", year:2024});
CREATE (mathstral:Model {name:"Mathstral", desc:"Mistral's mathematical reasoning model.", generation:"Mathstral", year:2024});

// xAI Models
CREATE (grok1:Model {name:"Grok-1", desc:"xAI's first model with real-time access.", generation:"Grok-1", year:2023});
CREATE (grok15:Model {name:"Grok-1.5", desc:"xAI's improved reasoning model.", generation:"Grok-1.5", year:2024});
CREATE (grok2:Model {name:"Grok-2", desc:"xAI's multimodal model.", generation:"Grok-2", year:2024});

// Perplexity Models
CREATE (perplexity_model:Model {name:"Perplexity LLM", desc:"Perplexity's search-optimized language model.", generation:"Perplexity", year:2024});

// Aleph Alpha Models
CREATE (luminous_base:Model {name:"Luminous Base", desc:"Aleph Alpha's foundational European LLM.", generation:"Luminous", year:2022});
CREATE (luminous_extended:Model {name:"Luminous Extended", desc:"Aleph Alpha's enhanced multilingual model.", generation:"Luminous", year:2023});
CREATE (luminous_supreme:Model {name:"Luminous Supreme", desc:"Aleph Alpha's most capable model.", generation:"Luminous", year:2023});

// Chinese Models (Expanded)
CREATE (wudao:Model {name:"WuDao 2.0", desc:"BAAI's trillion-parameter Chinese/English LLM.", generation:"WuDao", year:2021, parameters:"1.75T"});
CREATE (ernie:Model {name:"ERNIE Bot", desc:"Baidu's GPT-4 rival, Chinese NLP focus.", generation:"ERNIE", year:2023});
CREATE (ernie4:Model {name:"ERNIE 4.0", desc:"Baidu's latest multimodal model.", generation:"ERNIE-4", year:2023});
CREATE (tongyi:Model {name:"Tongyi Qianwen", desc:"Alibaba's Chinese LLM (original).", generation:"Qianwen", year:2023});
CREATE (qwen:Model {name:"Qwen", desc:"Alibaba Cloud's advanced Chinese LLM series.", generation:"Qwen", year:2023});
CREATE (qwen2:Model {name:"Qwen 2", desc:"Alibaba's improved multilingual model.", generation:"Qwen-2", year:2024});
CREATE (glm130b:Model {name:"GLM-130B", desc:"Zhipu's large open-source Chinese LLM.", generation:"GLM", year:2022, parameters:"130B"});
CREATE (glm4:Model {name:"GLM-4", desc:"Zhipu's latest Chinese LLM.", generation:"GLM-4", year:2024});
CREATE (glm45:Model {name:"GLM-4.5", desc:"Zhipu's enhanced reasoning model.", generation:"GLM-4.5", year:2024});
CREATE (chatglm:Model {name:"ChatGLM", desc:"Zhipu's conversational AI model.", type:"Interface", year:2023});
CREATE (deepseek_coder:Model {name:"DeepSeek Coder", desc:"DeepSeek's code generation model.", generation:"DeepSeek", year:2023});
CREATE (deepseek_v2:Model {name:"DeepSeek-V2", desc:"DeepSeek's mixture-of-experts model.", generation:"DeepSeek-V2", year:2024, parameters:"236B"});
CREATE (deepseek_v3:Model {name:"DeepSeek-V3", desc:"DeepSeek's latest large model.", generation:"DeepSeek-V3", year:2024, parameters:"671B"});
CREATE (deepseek_r1:Model {name:"DeepSeek R1", desc:"DeepSeek's reasoning-focused model.", generation:"DeepSeek-R1", year:2024});
CREATE (yi34b:Model {name:"Yi-34B", desc:"01.AI's large language model.", generation:"Yi", year:2023, parameters:"34B"});
CREATE (minimax_model:Model {name:"MiniMax LLM", desc:"MiniMax's multimodal text-to-video model.", generation:"MiniMax", year:2024});
CREATE (titan:Model {name:"Titan", desc:"Tencent's large language model.", generation:"Titan", year:2024});
CREATE (dreamwriter:Model {name:"Dreamwriter", desc:"Tencent's creative writing AI model.", generation:"Dreamwriter", year:2024});

// Japanese Models
CREATE (cristal:Model {name:"Cristal Intelligence", desc:"SoftBank-OpenAI enterprise agent for Japan.", generation:"Cristal", year:2025});
CREATE (sakana_model:Model {name:"Sakana AI Model", desc:"Collective intelligence-based AI system.", generation:"Sakana", year:2024});

// Korean Models
CREATE (exaone:Model {name:"ExaONE 4.0", desc:"LG's hybrid reasoning and language model.", generation:"ExaONE", year:2024});
CREATE (hyperclova:Model {name:"HyperClova X", desc:"Naver's Korean-optimized large language model.", generation:"HyperClova", year:2024});
CREATE (kt_model:Model {name:"KT Korean LLM", desc:"KT Corporation's Korean language model.", generation:"KT", year:2024});

// Indian Models
CREATE (sarvam_m:Model {name:"Sarvam M", desc:"Multilingual Indian LLM supporting Hindi, Tamil, Telugu, Kannada.", generation:"Sarvam", year:2024});
CREATE (project_indus:Model {name:"Project Indus", desc:"Tech Mahindra's Hindi-focused LLM (539M parameters).", generation:"Indus", year:2024, parameters:"539M"});
CREATE (tamil_llama:Model {name:"Tamil-LLAMA", desc:"Tamil language-specific model.", generation:"Tamil", year:2024});
CREATE (krutrim:Model {name:"Krutrim AI", desc:"Indian multilingual language model.", generation:"Krutrim", year:2024});
CREATE (openhathi:Model {name:"OpenHathi", desc:"Open-source Indian language model.", generation:"OpenHathi", year:2024});
CREATE (bhashini:Model {name:"Bhashini", desc:"Government-backed Indian language AI platform.", generation:"Bhashini", year:2024});

// Image Generation Models (Expanded)
CREATE (stable_diffusion:Model {name:"Stable Diffusion", desc:"Stability AI's open-source image generator.", type:"Image", year:2022});
CREATE (stable_diffusion_xl:Model {name:"Stable Diffusion XL", desc:"Stability AI's enhanced image model.", type:"Image", year:2023});
CREATE (midjourney_v1:Model {name:"Midjourney v1", desc:"Midjourney's first image generation model.", type:"Image", year:2022});
CREATE (midjourney_v6:Model {name:"Midjourney v6", desc:"Midjourney's latest image generation model.", type:"Image", year:2024});

// ==== PEOPLE TO COMPANIES (Extended Relationships) ====
// Academic foundations
MATCH (turing:Person {name:"Alan Turing"}), (mit:Org {name:"MIT AI Lab"})
CREATE (turing)-[:INSPIRED]->(mit);
MATCH (mccarthy:Person {name:"John McCarthy"}), (mit:Org {name:"MIT AI Lab"})
CREATE (mccarthy)-[:FOUNDED]->(mit);
MATCH (mccarthy:Person {name:"John McCarthy"}), (stanford:Org {name:"Stanford AI Lab"})
CREATE (mccarthy)-[:FOUNDED]->(stanford);
MATCH (minsky:Person {name:"Marvin Minsky"}), (mit:Org {name:"MIT AI Lab"})
CREATE (minsky)-[:COFOUNDED]->(mit);
MATCH (newell:Person {name:"Allen Newell"}), (cmu:Org {name:"Carnegie Mellon AI"})
CREATE (newell)-[:FOUNDED]->(cmu);
MATCH (simon:Person {name:"Herbert Simon"}), (cmu:Org {name:"Carnegie Mellon AI"})
CREATE (simon)-[:COFOUNDED]->(cmu);

// Deep learning pioneers
MATCH (lecun:Person {name:"Yann LeCun"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (lecun)-[:CHIEF_SCIENTIST]->(meta);
MATCH (bengio:Person {name:"Yoshua Bengio"}), (montreal:Org {name:"MILA Montreal"})
CREATE (bengio)-[:FOUNDED]->(montreal);
MATCH (hinton:Person {name:"Geoffrey Hinton"}), (toronto:Org {name:"University of Toronto"})
CREATE (hinton)-[:PROFESSOR_AT]->(toronto);
MATCH (hinton:Person {name:"Geoffrey Hinton"}), (googlebrain:Org {name:"Google Brain"})
CREATE (hinton)-[:JOINED]->(googlebrain);
MATCH (schmidhuber:Person {name:"Jürgen Schmidhuber"}), (ethz:Org {name:"ETH Zurich AI"})
CREATE (schmidhuber)-[:AFFILIATED_WITH]->(ethz);

// Google ecosystem
MATCH (dean:Person {name:"Jeff Dean"}), (googlebrain:Org {name:"Google Brain"})
CREATE (dean)-[:COFOUNDED]->(googlebrain);
MATCH (ng:Person {name:"Andrew Ng"}), (googlebrain:Org {name:"Google Brain"})
CREATE (ng)-[:COFOUNDED]->(googlebrain);
MATCH (norvig:Person {name:"Peter Norvig"}), (googlebrain:Org {name:"Google Brain"})
CREATE (norvig)-[:RESEARCH_DIRECTOR]->(googlebrain);

// OpenAI comprehensive relationships
MATCH (altman:Person {name:"Sam Altman"}), (openai:Org {name:"OpenAI"})
CREATE (altman)-[:CEO]->(openai);
MATCH (sutskever:Person {name:"Ilya Sutskever"}), (openai:Org {name:"OpenAI"})
CREATE (sutskever)-[:CHIEF_SCIENTIST {left:2024}]->(openai);
MATCH (brockman:Person {name:"Greg Brockman"}), (openai:Org {name:"OpenAI"})
CREATE (brockman)-[:PRESIDENT]->(openai);
MATCH (schulman:Person {name:"John Schulman"}), (openai:Org {name:"OpenAI"})
CREATE (schulman)-[:COFOUNDER {left:2024}]->(openai);
MATCH (zaremba:Person {name:"Wojciech Zaremba"}), (openai:Org {name:"OpenAI"})
CREATE (zaremba)-[:COFOUNDER]->(openai);
MATCH (chen:Person {name:"Mark Chen"}), (openai:Org {name:"OpenAI"})
CREATE (chen)-[:RESEARCH_SCIENTIST]->(openai);
MATCH (radford:Person {name:"Alec Radford"}), (openai:Org {name:"OpenAI"})
CREATE (radford)-[:RESEARCH_SCIENTIST]->(openai);
MATCH (karpathy:Person {name:"Andrej Karpathy"}), (openai:Org {name:"OpenAI"})
CREATE (karpathy)-[:FOUNDING_MEMBER {left:2022, rejoined:2023, left_again:2024}]->(openai);
MATCH (musk:Person {name:"Elon Musk"}), (openai:Org {name:"OpenAI"})
CREATE (musk)-[:COFOUNDER {left:2018}]->(openai);

// Anthropic relationships
MATCH (dario:Person {name:"Dario Amodei"}), (anthropic:Org {name:"Anthropic"})
CREATE (dario)-[:CEO]->(anthropic);
MATCH (daniela:Person {name:"Daniela Amodei"}), (anthropic:Org {name:"Anthropic"})
CREATE (daniela)-[:PRESIDENT]->(anthropic);
MATCH (leike:Person {name:"Jan Leike"}), (anthropic:Org {name:"Anthropic"})  
CREATE (leike)-[:HEAD_OF_ALIGNMENT {from:"OpenAI", joined:2024}]->(anthropic);
MATCH (christiano:Person {name:"Paul Christiano"}), (anthropic:Org {name:"Anthropic"})
CREATE (christiano)-[:ADVISOR]->(anthropic);

// DeepMind comprehensive
MATCH (hassabis:Person {name:"Demis Hassabis"}), (deepmind:Org {name:"DeepMind"})
CREATE (hassabis)-[:CEO]->(deepmind);
MATCH (suleyman:Person {name:"Mustafa Suleyman"}), (deepmind:Org {name:"DeepMind"})
CREATE (suleyman)-[:COFOUNDER {left:2022}]->(deepmind);
MATCH (legg:Person {name:"Shane Legg"}), (deepmind:Org {name:"DeepMind"})
CREATE (legg)-[:CHIEF_SCIENTIST]->(deepmind);

// Transformer inventors
MATCH (vaswani:Person {name:"Ashish Vaswani"}), (googlebrain:Org {name:"Google Brain"})
CREATE (vaswani)-[:RESEARCH_SCIENTIST {left:2021}]->(googlebrain);
MATCH (shazeer:Person {name:"Noam Shazeer"}), (googlebrain:Org {name:"Google Brain"})
CREATE (shazeer)-[:RESEARCH_SCIENTIST {left:2021}]->(googlebrain);
MATCH (shazeer:Person {name:"Noam Shazeer"}), (characterai:Org {name:"Character.AI"})
CREATE (shazeer)-[:COFOUNDER]->(characterai);
MATCH (gomez:Person {name:"Aidan Gomez"}), (googlebrain:Org {name:"Google Brain"})
CREATE (gomez)-[:RESEARCH_INTERN {left:2017}]->(googlebrain);
MATCH (gomez:Person {name:"Aidan Gomez"}), (cohere:Org {name:"Cohere"})
CREATE (gomez)-[:CEO]->(cohere);
MATCH (frosst:Person {name:"Nick Frosst"}), (googlebrain:Org {name:"Google Brain"})
CREATE (frosst)-[:RESEARCH_SCIENTIST {left:2021}]->(googlebrain);
MATCH (frosst:Person {name:"Nick Frosst"}), (cohere:Org {name:"Cohere"})
CREATE (frosst)-[:COFOUNDER]->(cohere);
MATCH (jones:Person {name:"Llion Jones"}), (googlebrain:Org {name:"Google Brain"})
CREATE (jones)-[:RESEARCH_SCIENTIST {left:2023}]->(googlebrain);
MATCH (jones:Person {name:"Llion Jones"}), (sakana:Org {name:"Sakana AI"})
CREATE (jones)-[:COFOUNDER]->(sakana);

// New generation startups
MATCH (mensch:Person {name:"Arthur Mensch"}), (mistral:Org {name:"Mistral AI"})
CREATE (mensch)-[:CEO]->(mistral);
MATCH (lample:Person {name:"Guillaume Lample"}), (mistral:Org {name:"Mistral AI"})
CREATE (lample)-[:COFOUNDER]->(mistral);
MATCH (lacroix:Person {name:"Timothée Lacroix"}), (mistral:Org {name:"Mistral AI"})
CREATE (lacroix)-[:COFOUNDER]->(mistral);
MATCH (holz:Person {name:"David Holz"}), (midjourney:Org {name:"Midjourney"})
CREATE (holz)-[:FOUNDER]->(midjourney);
MATCH (mostaque:Person {name:"Emad Mostaque"}), (stability:Org {name:"Stability AI"})
CREATE (mostaque)-[:FOUNDER {left:2024}]->(stability);
MATCH (musk:Person {name:"Elon Musk"}), (xai:Org {name:"xAI"})
CREATE (musk)-[:FOUNDER]->(xai);

// Microsoft connections
MATCH (suleyman:Person {name:"Mustafa Suleyman"}), (microsoft:Org {name:"Microsoft Research"})
CREATE (suleyman)-[:EVP_COPILOT {joined:2024}]->(microsoft);

// Japanese ecosystem
MATCH (ha:Person {name:"David Ha"}), (sakana:Org {name:"Sakana AI"})
CREATE (ha)-[:COFOUNDER]->(sakana);
MATCH (son:Person {name:"Masayoshi Son"}), (softbank:Org {name:"SoftBank"})
CREATE (son)-[:FOUNDER]->(softbank);
MATCH (softbank:Org {name:"SoftBank"}), (sb_openai:Org {name:"SB OpenAI Japan"})
CREATE (softbank)-[:JOINT_VENTURE_WITH]->(sb_openai);

// Indian ecosystem
MATCH (srinivasan:Person {name:"Vivek Srinivasan"}), (sarvam:Org {name:"Sarvam AI"})
CREATE (srinivasan)-[:COFOUNDER]->(sarvam);
MATCH (khandelwal:Person {name:"Pratyush Khandelwal"}), (sarvam:Org {name:"Sarvam AI"})
CREATE (khandelwal)-[:COFOUNDER]->(sarvam);

// Korean ecosystem
MATCH (kim:Person {name:"Dr. Kim"}), (lg_ai:Org {name:"LG AI Research"})
CREATE (kim)-[:RESEARCH_DIRECTOR]->(lg_ai);
MATCH (lee:Person {name:"Hae-Jin Lee"}), (naver:Org {name:"Naver Corporation"})
CREATE (lee)-[:AI_DIRECTOR]->(naver);

// Chinese ecosystem
MATCH (robinli:Person {name:"Robin Li"}), (baidu:Org {name:"Baidu"})
CREATE (robinli)-[:CEO]->(baidu);
MATCH (feifei:Person {name:"Fei-Fei Li"}), (stanford:Org {name:"Stanford AI Lab"})
CREATE (feifei)-[:PROFESSOR]->(stanford);
MATCH (xuli:Person {name:"Xu Li"}), (sensetime:Org {name:"SenseTime"})
CREATE (xuli)-[:COFOUNDER]->(sensetime);
MATCH (wangxg:Person {name:"Wang Xiaogang"}), (sensetime:Org {name:"SenseTime"})
CREATE (wangxg)-[:COFOUNDER]->(sensetime);
MATCH (tang:Person {name:"Tang Xiao'ou"}), (sensetime:Org {name:"SenseTime"})
CREATE (tang)-[:COFOUNDER]->(sensetime);
MATCH (zhengkai:Person {name:"Kai-Fu Lee"}), (sinovation:Org {name:"Sinovation Ventures"})
CREATE (zhengkai)-[:FOUNDER]->(sinovation);
MATCH (zhengkai:Person {name:"Kai-Fu Lee"}), (01ai:Org {name:"01.AI"})
CREATE (zhengkai)-[:FOUNDER]->(01ai);
MATCH (liangw:Person {name:"Liang Wenfeng"}), (deepseek:Org {name:"DeepSeek"})
CREATE (liangw)-[:FOUNDER]->(deepseek);
MATCH (yangqing:Person {name:"Yang Qing"}), (alibaba:Org {name:"Alibaba DAMO"})
CREATE (yangqing)-[:CHIEF_SCIENTIST]->(alibaba);

// ==== COMPANY TO MODEL RELATIONSHIPS (Comprehensive) ====
// OpenAI model releases
MATCH (openai:Org {name:"OpenAI"}), (gpt1:Model {name:"GPT-1"})
CREATE (openai)-[:RELEASED]->(gpt1);
MATCH (openai:Org {name:"OpenAI"}), (gpt2:Model {name:"GPT-2"})
CREATE (openai)-[:RELEASED]->(gpt2);
MATCH (openai:Org {name:"OpenAI"}), (gpt3:Model {name:"GPT-3"})
CREATE (openai)-[:RELEASED]->(gpt3);
MATCH (openai:Org {name:"OpenAI"}), (gpt35:Model {name:"GPT-3.5"})
CREATE (openai)-[:RELEASED]->(gpt35);
MATCH (openai:Org {name:"OpenAI"}), (gpt4:Model {name:"GPT-4"})
CREATE (openai)-[:RELEASED]->(gpt4);
MATCH (openai:Org {name:"OpenAI"}), (gpt4o:Model {name:"GPT-4o"})
CREATE (openai)-[:RELEASED]->(gpt4o);
MATCH (openai:Org {name:"OpenAI"}), (gpt5:Model {name:"GPT-5"})
CREATE (openai)-[:DEVELOPING]->(gpt5);
MATCH (openai:Org {name:"OpenAI"}), (o1:Model {name:"o1"})
CREATE (openai)-[:RELEASED]->(o1);
MATCH (openai:Org {name:"OpenAI"}), (o3:Model {name:"o3"})
CREATE (openai)-[:RELEASED]->(o3);
MATCH (openai:Org {name:"OpenAI"}), (dalle:Model {name:"DALL-E"})
CREATE (openai)-[:RELEASED]->(dalle);
MATCH (openai:Org {name:"OpenAI"}), (dalle2:Model {name:"DALL-E 2"})
CREATE (openai)-[:RELEASED]->(dalle2);
MATCH (openai:Org {name:"OpenAI"}), (dalle3:Model {name:"DALL-E 3"})
CREATE (openai)-[:RELEASED]->(dalle3);
MATCH (openai:Org {name:"OpenAI"}), (sora:Model {name:"Sora"})
CREATE (openai)-[:RELEASED]->(sora);
MATCH (openai:Org {name:"OpenAI"}), (chatgpt:Model {name:"ChatGPT"})
CREATE (openai)-[:RELEASED]->(chatgpt);

// Microsoft model releases
MATCH (microsoft_copilot:Org {name:"Microsoft Copilot"}), (prometheus:Model {name:"Prometheus"})
CREATE (microsoft_copilot)-[:RELEASED]->(prometheus);
MATCH (microsoft_copilot:Org {name:"Microsoft Copilot"}), (copilot_gpt4:Model {name:"Copilot GPT-4"})
CREATE (microsoft_copilot)-[:RELEASED]->(copilot_gpt4);

// Anthropic model releases
MATCH (anthropic:Org {name:"Anthropic"}), (claude1:Model {name:"Claude 1"})
CREATE (anthropic)-[:RELEASED]->(claude1);
MATCH (anthropic:Org {name:"Anthropic"}), (claude2:Model {name:"Claude 2"})
CREATE (anthropic)-[:RELEASED]->(claude2);
MATCH (anthropic:Org {name:"Anthropic"}), (claude3_haiku:Model {name:"Claude 3 Haiku"})
CREATE (anthropic)-[:RELEASED]->(claude3_haiku);
MATCH (anthropic:Org {name:"Anthropic"}), (claude3_sonnet:Model {name:"Claude 3 Sonnet"})
CREATE (anthropic)-[:RELEASED]->(claude3_sonnet);
MATCH (anthropic:Org {name:"Anthropic"}), (claude3_opus:Model {name:"Claude 3 Opus"})
CREATE (anthropic)-[:RELEASED]->(claude3_opus);
MATCH (anthropic:Org {name:"Anthropic"}), (claude35_sonnet:Model {name:"Claude 3.5 Sonnet"})
CREATE (anthropic)-[:RELEASED]->(claude35_sonnet);

// Google/DeepMind model releases
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (lamda:Model {name:"LaMDA"})
CREATE (googledeepmind)-[:RELEASED]->(lamda);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (palm:Model {name:"PaLM"})
CREATE (googledeepmind)-[:RELEASED]->(palm);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (palm2:Model {name:"PaLM 2"})
CREATE (googledeepmind)-[:RELEASED]->(palm2);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (gemini_nano:Model {name:"Gemini Nano"})
CREATE (googledeepmind)-[:RELEASED]->(gemini_nano);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (gemini_pro:Model {name:"Gemini Pro"})
CREATE (googledeepmind)-[:RELEASED]->(gemini_pro);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (gemini_ultra:Model {name:"Gemini Ultra"})
CREATE (googledeepmind)-[:RELEASED]->(gemini_ultra);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (gemini_15_pro:Model {name:"Gemini 1.5 Pro"})
CREATE (googledeepmind)-[:RELEASED]->(gemini_15_pro);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (bard:Model {name:"Bard"})
CREATE (googledeepmind)-[:RELEASED]->(bard);
MATCH (deepmind:Org {name:"DeepMind"}), (alphago:Model {name:"AlphaGo"})
CREATE (deepmind)-[:RELEASED]->(alphago);
MATCH (deepmind:Org {name:"DeepMind"}), (alphazero:Model {name:"AlphaZero"})
CREATE (deepmind)-[:RELEASED]->(alphazero);
MATCH (deepmind:Org {name:"DeepMind"}), (alphafold:Model {name:"AlphaFold"})
CREATE (deepmind)-[:RELEASED]->(alphafold);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (alphafold3:Model {name:"AlphaFold 3"})
CREATE (googledeepmind)-[:RELEASED]->(alphafold3);

// Meta model releases
MATCH (meta:Org {name:"Meta AI (FAIR)"}), (llama:Model {name:"LLaMA"})
CREATE (meta)-[:RELEASED]->(llama);
MATCH (meta:Org {name:"Meta AI (FAIR)"}), (llama2:Model {name:"Llama 2"})
CREATE (meta)-[:RELEASED]->(llama2);
MATCH (meta:Org {name:"Meta AI (FAIR)"}), (llama3:Model {name:"Llama 3"})
CREATE (meta)-[:RELEASED]->(llama3);
MATCH (meta:Org {name:"Meta AI (FAIR)"}), (codellama:Model {name:"Code Llama"})
CREATE (meta)-[:RELEASED]->(codellama);

// Cohere model releases
MATCH (cohere:Org {name:"Cohere"}), (command:Model {name:"Command"})
CREATE (cohere)-[:RELEASED]->(command);
MATCH (cohere:Org {name:"Cohere"}), (command_r:Model {name:"Command R"})
CREATE (cohere)-[:RELEASED]->(command_r);
MATCH (cohere:Org {name:"Cohere"}), (aya_vision:Model {name:"Aya Vision"})
CREATE (cohere)-[:RELEASED]->(aya_vision);

// Mistral model releases
MATCH (mistral:Org {name:"Mistral AI"}), (mistral7b:Model {name:"Mistral 7B"})
CREATE (mistral)-[:RELEASED]->(mistral7b);
MATCH (mistral:Org {name:"Mistral AI"}), (mixtral:Model {name:"Mixtral 8x7B"})
CREATE (mistral)-[:RELEASED]->(mixtral);
MATCH (mistral:Org {name:"Mistral AI"}), (mistral_large:Model {name:"Mistral Large"})
CREATE (mistral)-[:RELEASED]->(mistral_large);
MATCH (mistral:Org {name:"Mistral AI"}), (codestral:Model {name:"Codestral"})
CREATE (mistral)-[:RELEASED]->(codestral);
MATCH (mistral:Org {name:"Mistral AI"}), (mathstral:Model {name:"Mathstral"})
CREATE (mistral)-[:RELEASED]->(mathstral);

// xAI model releases
MATCH (xai:Org {name:"xAI"}), (grok1:Model {name:"Grok-1"})
CREATE (xai)-[:RELEASED]->(grok1);
MATCH (xai:Org {name:"xAI"}), (grok15:Model {name:"Grok-1.5"})
CREATE (xai)-[:RELEASED]->(grok15);
MATCH (xai:Org {name:"xAI"}), (grok2:Model {name:"Grok-2"})
CREATE (xai)-[:RELEASED]->(grok2);

// Perplexity model releases
MATCH (perplexity:Org {name:"Perplexity AI"}), (perplexity_model:Model {name:"Perplexity LLM"})
CREATE (perplexity)-[:RELEASED]->(perplexity_model);

// Aleph Alpha model releases
MATCH (aleph_alpha:Org {name:"Aleph Alpha"}), (luminous_base:Model {name:"Luminous Base"})
CREATE (aleph_alpha)-[:RELEASED]->(luminous_base);
MATCH (aleph_alpha:Org {name:"Aleph Alpha"}), (luminous_extended:Model {name:"Luminous Extended"})
CREATE (aleph_alpha)-[:RELEASED]->(luminous_extended);
MATCH (aleph_alpha:Org {name:"Aleph Alpha"}), (luminous_supreme:Model {name:"Luminous Supreme"})
CREATE (aleph_alpha)-[:RELEASED]->(luminous_supreme);

// Chinese model releases (expanded)
MATCH (baai:Org {name:"Beijing Academy of AI (BAAI)"}), (wudao:Model {name:"WuDao 2.0"})
CREATE (baai)-[:RELEASED]->(wudao);
MATCH (baidu:Org {name:"Baidu"}), (ernie:Model {name:"ERNIE Bot"})
CREATE (baidu)-[:RELEASED]->(ernie);
MATCH (baidu:Org {name:"Baidu"}), (ernie4:Model {name:"ERNIE 4.0"})
CREATE (baidu)-[:RELEASED]->(ernie4);
MATCH (alibaba:Org {name:"Alibaba DAMO"}), (tongyi:Model {name:"Tongyi Qianwen"})
CREATE (alibaba)-[:RELEASED]->(tongyi);
MATCH (alibaba_cloud:Org {name:"Alibaba Cloud"}), (qwen:Model {name:"Qwen"})
CREATE (alibaba_cloud)-[:RELEASED]->(qwen);
MATCH (alibaba_cloud:Org {name:"Alibaba Cloud"}), (qwen2:Model {name:"Qwen 2"})
CREATE (alibaba_cloud)-[:RELEASED]->(qwen2);
MATCH (zhipu:Org {name:"Zhipu AI"}), (glm130b:Model {name:"GLM-130B"})
CREATE (zhipu)-[:RELEASED]->(glm130b);
MATCH (zhipu:Org {name:"Zhipu AI"}), (glm4:Model {name:"GLM-4"})
CREATE (zhipu)-[:RELEASED]->(glm4);
MATCH (zhipu:Org {name:"Zhipu AI"}), (glm45:Model {name:"GLM-4.5"})
CREATE (zhipu)-[:RELEASED]->(glm45);
MATCH (zhipu:Org {name:"Zhipu AI"}), (chatglm:Model {name:"ChatGLM"})
CREATE (zhipu)-[:RELEASED]->(chatglm);
MATCH (deepseek:Org {name:"DeepSeek"}), (deepseek_coder:Model {name:"DeepSeek Coder"})
CREATE (deepseek)-[:RELEASED]->(deepseek_coder);
MATCH (deepseek:Org {name:"DeepSeek"}), (deepseek_v2:Model {name:"DeepSeek-V2"})
CREATE (deepseek)-[:RELEASED]->(deepseek_v2);
MATCH (deepseek:Org {name:"DeepSeek"}), (deepseek_v3:Model {name:"DeepSeek-V3"})
CREATE (deepseek)-[:RELEASED]->(deepseek_v3);
MATCH (deepseek:Org {name:"DeepSeek"}), (deepseek_r1:Model {name:"DeepSeek R1"})
CREATE (deepseek)-[:RELEASED]->(deepseek_r1);
MATCH (01ai:Org {name:"01.AI"}), (yi34b:Model {name:"Yi-34B"})
CREATE (01ai)-[:RELEASED]->(yi34b);
MATCH (minimax:Org {name:"MiniMax"}), (minimax_model:Model {name:"MiniMax LLM"})
CREATE (minimax)-[:RELEASED]->(minimax_model);
MATCH (tencent:Org {name:"Tencent AI Lab"}), (titan:Model {name:"Titan"})
CREATE (tencent)-[:RELEASED]->(titan);
MATCH (tencent:Org {name:"Tencent AI Lab"}), (dreamwriter:Model {name:"Dreamwriter"})
CREATE (tencent)-[:RELEASED]->(dreamwriter);

// Japanese model releases
MATCH (sb_openai:Org {name:"SB OpenAI Japan"}), (cristal:Model {name:"Cristal Intelligence"})
CREATE (sb_openai)-[:RELEASED]->(cristal);
MATCH (sakana:Org {name:"Sakana AI"}), (sakana_model:Model {name:"Sakana AI Model"})
CREATE (sakana)-[:RELEASED]->(sakana_model);

// Korean model releases
MATCH (lg_ai:Org {name:"LG AI Research"}), (exaone:Model {name:"ExaONE 4.0"})
CREATE (lg_ai)-[:RELEASED]->(exaone);
MATCH (naver:Org {name:"Naver Corporation"}), (hyperclova:Model {name:"HyperClova X"})
CREATE (naver)-[:RELEASED]->(hyperclova);
MATCH (kt:Org {name:"KT Corporation"}), (kt_model:Model {name:"KT Korean LLM"})
CREATE (kt)-[:RELEASED]->(kt_model);

// Indian model releases
MATCH (sarvam:Org {name:"Sarvam AI"}), (sarvam_m:Model {name:"Sarvam M"})
CREATE (sarvam)-[:RELEASED]->(sarvam_m);
MATCH (tech_mahindra:Org {name:"Tech Mahindra"}), (project_indus:Model {name:"Project Indus"})
CREATE (tech_mahindra)-[:RELEASED]->(project_indus);

// Image generation models
MATCH (stability:Org {name:"Stability AI"}), (stable_diffusion:Model {name:"Stable Diffusion"})
CREATE (stability)-[:RELEASED]->(stable_diffusion);
MATCH (stability:Org {name:"Stability AI"}), (stable_diffusion_xl:Model {name:"Stable Diffusion XL"})
CREATE (stability)-[:RELEASED]->(stable_diffusion_xl);
MATCH (midjourney:Org {name:"Midjourney"}), (midjourney_v1:Model {name:"Midjourney v1"})
CREATE (midjourney)-[:RELEASED]->(midjourney_v1);
MATCH (midjourney:Org {name:"Midjourney"}), (midjourney_v6:Model {name:"Midjourney v6"})
CREATE (midjourney)-[:RELEASED]->(midjourney_v6);

// ==== COMPLEX RELATIONSHIPS (Expanded) ====
// Academic lineage and mentorship
MATCH (hinton:Person {name:"Geoffrey Hinton"}), (sutskever:Person {name:"Ilya Sutskever"})
CREATE (hinton)-[:MENTORED]->(sutskever);
MATCH (hinton:Person {name:"Geoffrey Hinton"}), (krizhevsky:Person {name:"Alex Krizhevsky"})
CREATE (hinton)-[:MENTORED]->(krizhevsky);
MATCH (lecun:Person {name:"Yann LeCun"}), (lample:Person {name:"Guillaume Lample"})
CREATE (lecun)-[:MENTORED]->(lample);
MATCH (bengio:Person {name:"Yoshua Bengio"}), (gomez:Person {name:"Aidan Gomez"})
CREATE (bengio)-[:MENTORED]->(gomez);
MATCH (ng:Person {name:"Andrew Ng"}), (chen:Person {name:"Mark Chen"})
CREATE (ng)-[:MENTORED]->(chen);

// Company evolution and acquisitions
MATCH (googlebrain:Org {name:"Google Brain"}), (deepmind:Org {name:"DeepMind"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (googlebrain)-[:MERGED_WITH]->(deepmind),
       (deepmind)-[:BECAME]->(googledeepmind),
       (googlebrain)-[:BECAME]->(googledeepmind);
MATCH (openai:Org {name:"OpenAI"}), (anthropic:Org {name:"Anthropic"})
CREATE (openai)-[:SPAWNED]->(anthropic);
MATCH (inflection:Org {name:"Inflection AI"}), (microsoft:Org {name:"Microsoft Research"})
CREATE (inflection)-[:TALENT_ACQUIRED_BY]->(microsoft);
MATCH (openai:Org {name:"OpenAI"}), (sb_openai:Org {name:"SB OpenAI Japan"})
CREATE (openai)-[:JOINT_VENTURE]->(sb_openai);

// Investment relationships (detailed)
MATCH (microsoft:Org {name:"Microsoft Research"}), (openai:Org {name:"OpenAI"})
CREATE (microsoft)-[:INVESTED_IN {amount:"$13B", years:"2019-2024", equity:"49%"}]->(openai);
MATCH (amazon:Org {name:"Amazon AI"}), (anthropic:Org {name:"Anthropic"})
CREATE (amazon)-[:INVESTED_IN {amount:"$4B", year:2023}]->(anthropic);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (anthropic:Org {name:"Anthropic"})
CREATE (googledeepmind)-[:INVESTED_IN {amount:"$300M", year:2022}]->(anthropic);
MATCH (a16z:Org {name:"Andreessen Horowitz"}), (mistral:Org {name:"Mistral AI"})
CREATE (a16z)-[:INVESTED_IN {amount:"$415M", year:2023}]->(mistral);
MATCH (sequoia:Org {name:"Sequoia Capital"}), (openai:Org {name:"OpenAI"})
CREATE (sequoia)-[:INVESTED_IN {amount:"Multiple rounds"}]->(openai);
MATCH (khosla:Org {name:"Khosla Ventures"}), (openai:Org {name:"OpenAI"})
CREATE (khosla)-[:INVESTED_IN {amount:"Early investor"}]->(openai);
MATCH (sinovation:Org {name:"Sinovation Ventures"}), (01ai:Org {name:"01.AI"})
CREATE (sinovation)-[:INVESTED_IN {amount:"$200M", year:2023}]->(01ai);
MATCH (softbank:Org {name:"SoftBank"}), (perplexity:Org {name:"Perplexity AI"})
CREATE (softbank)-[:INVESTED_IN {amount:"$500M", year:2024}]->(perplexity);

// Hardware dependencies and partnerships
MATCH (nvidia:Org {name:"NVIDIA AI"}), (openai:Org {name:"OpenAI"})
CREATE (nvidia)-[:PROVIDES_HARDWARE]->(openai);
MATCH (nvidia:Org {name:"NVIDIA AI"}), (anthropic:Org {name:"Anthropic"})
CREATE (nvidia)-[:PROVIDES_HARDWARE]->(anthropic);
MATCH (nvidia:Org {name:"NVIDIA AI"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (nvidia)-[:PROVIDES_HARDWARE]->(meta);
MATCH (nvidia:Org {name:"NVIDIA AI"}), (xai:Org {name:"xAI"})
CREATE (nvidia)-[:PROVIDES_HARDWARE]->(xai);
MATCH (groq:Org {name:"Groq"}), (perplexity:Org {name:"Perplexity AI"})
CREATE (groq)-[:PROVIDES_INFERENCE]->(perplexity);
MATCH (cerebras:Org {name:"Cerebras"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (cerebras)-[:PROVIDES_TRAINING]->(meta);
MATCH (sambanova:Org {name:"SambaNova"}), (anthropic:Org {name:"Anthropic"})
CREATE (sambanova)-[:PROVIDES_TRAINING]->(anthropic);

// Competitive relationships (expanded)
MATCH (openai:Org {name:"OpenAI"}), (anthropic:Org {name:"Anthropic"})
CREATE (openai)-[:COMPETES_WITH]->(anthropic);
MATCH (openai:Org {name:"OpenAI"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (openai)-[:COMPETES_WITH]->(googledeepmind);
MATCH (openai:Org {name:"OpenAI"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (openai)-[:COMPETES_WITH]->(meta);
MATCH (openai:Org {name:"OpenAI"}), (xai:Org {name:"xAI"})
CREATE (openai)-[:COMPETES_WITH]->(xai);
MATCH (anthropic:Org {name:"Anthropic"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (anthropic)-[:COMPETES_WITH]->(googledeepmind);
MATCH (chatgpt:Model {name:"ChatGPT"}), (claude35_sonnet:Model {name:"Claude 3.5 Sonnet"})
CREATE (chatgpt)-[:COMPETES_WITH]->(claude35_sonnet);
MATCH (gpt4o:Model {name:"GPT-4o"}), (gemini_15_pro:Model {name:"Gemini 1.5 Pro"})
CREATE (gpt4o)-[:COMPETES_WITH]->(gemini_15_pro);
MATCH (gpt4:Model {name:"GPT-4"}), (gpt4o:Model {name:"GPT-4o"})
CREATE (gpt4)-[:EVOLVED_TO]->(gpt4o);
MATCH (gpt4o:Model {name:"GPT-4o"}), (o1:Model {name:"o1"})
CREATE (gpt4o)-[:EVOLVED_TO]->(o1);
MATCH (o1:Model {name:"o1"}), (o3:Model {name:"o3"})
CREATE (o1)-[:EVOLVED_TO]->(o3);
MATCH (gpt4:Model {name:"GPT-4"}), (gpt5:Model {name:"GPT-5"})
CREATE (gpt4)-[:EVOLVED_TO]->(gpt5);

// Claude evolution
MATCH (claude1:Model {name:"Claude 1"}), (claude2:Model {name:"Claude 2"})
CREATE (claude1)-[:EVOLVED_TO]->(claude2);
MATCH (claude2:Model {name:"Claude 2"}), (claude3_haiku:Model {name:"Claude 3 Haiku"})
CREATE (claude2)-[:EVOLVED_TO]->(claude3_haiku);
MATCH (claude2:Model {name:"Claude 2"}), (claude3_sonnet:Model {name:"Claude 3 Sonnet"})
CREATE (claude2)-[:EVOLVED_TO]->(claude3_sonnet);
MATCH (claude2:Model {name:"Claude 2"}), (claude3_opus:Model {name:"Claude 3 Opus"})
CREATE (claude2)-[:EVOLVED_TO]->(claude3_opus);
MATCH (claude3_sonnet:Model {name:"Claude 3 Sonnet"}), (claude35_sonnet:Model {name:"Claude 3.5 Sonnet"})
CREATE (claude3_sonnet)-[:EVOLVED_TO]->(claude35_sonnet);

// Meta model evolution
MATCH (llama:Model {name:"LLaMA"}), (llama2:Model {name:"Llama 2"})
CREATE (llama)-[:EVOLVED_TO]->(llama2);
MATCH (llama2:Model {name:"Llama 2"}), (llama3:Model {name:"Llama 3"})
CREATE (llama2)-[:EVOLVED_TO]->(llama3);
MATCH (llama2:Model {name:"Llama 2"}), (codellama:Model {name:"Code Llama"})
CREATE (llama2)-[:SPECIALIZED_TO]->(codellama);

// Google model evolution
MATCH (lamda:Model {name:"LaMDA"}), (palm:Model {name:"PaLM"})
CREATE (lamda)-[:EVOLVED_TO]->(palm);
MATCH (palm:Model {name:"PaLM"}), (palm2:Model {name:"PaLM 2"})
CREATE (palm)-[:EVOLVED_TO]->(palm2);
MATCH (palm2:Model {name:"PaLM 2"}), (gemini_pro:Model {name:"Gemini Pro"})
CREATE (palm2)-[:EVOLVED_TO]->(gemini_pro);
MATCH (gemini_pro:Model {name:"Gemini Pro"}), (gemini_15_pro:Model {name:"Gemini 1.5 Pro"})
CREATE (gemini_pro)-[:EVOLVED_TO]->(gemini_15_pro);

// xAI model evolution
MATCH (grok1:Model {name:"Grok-1"}), (grok15:Model {name:"Grok-1.5"})
CREATE (grok1)-[:EVOLVED_TO]->(grok15);
MATCH (grok15:Model {name:"Grok-1.5"}), (grok2:Model {name:"Grok-2"})
CREATE (grok15)-[:EVOLVED_TO]->(grok2);

// Chinese model evolution
MATCH (glm130b:Model {name:"GLM-130B"}), (glm4:Model {name:"GLM-4"})
CREATE (glm130b)-[:EVOLVED_TO]->(glm4);
MATCH (glm4:Model {name:"GLM-4"}), (glm45:Model {name:"GLM-4.5"})
CREATE (glm4)-[:EVOLVED_TO]->(glm45);
MATCH (tongyi:Model {name:"Tongyi Qianwen"}), (qwen:Model {name:"Qwen"})
CREATE (tongyi)-[:EVOLVED_TO]->(qwen);
MATCH (qwen:Model {name:"Qwen"}), (qwen2:Model {name:"Qwen 2"})
CREATE (qwen)-[:EVOLVED_TO]->(qwen2);
MATCH (deepseek_v2:Model {name:"DeepSeek-V2"}), (deepseek_v3:Model {name:"DeepSeek-V3"})
CREATE (deepseek_v2)-[:EVOLVED_TO]->(deepseek_v3);
MATCH (deepseek_v3:Model {name:"DeepSeek-V3"}), (deepseek_r1:Model {name:"DeepSeek R1"})
CREATE (deepseek_v3)-[:EVOLVED_TO]->(deepseek_r1);
MATCH (ernie:Model {name:"ERNIE Bot"}), (ernie4:Model {name:"ERNIE 4.0"})
CREATE (ernie)-[:EVOLVED_TO]->(ernie4);

// Image generation evolution
MATCH (dalle:Model {name:"DALL-E"}), (dalle2:Model {name:"DALL-E 2"})
CREATE (dalle)-[:EVOLVED_TO]->(dalle2);
MATCH (dalle2:Model {name:"DALL-E 2"}), (dalle3:Model {name:"DALL-E 3"})
CREATE (dalle2)-[:EVOLVED_TO]->(dalle3);
MATCH (stable_diffusion:Model {name:"Stable Diffusion"}), (stable_diffusion_xl:Model {name:"Stable Diffusion XL"})
CREATE (stable_diffusion)-[:EVOLVED_TO]->(stable_diffusion_xl);
MATCH (midjourney_v1:Model {name:"Midjourney v1"}), (midjourney_v6:Model {name:"Midjourney v6"})
CREATE (midjourney_v1)-[:EVOLVED_TO]->(midjourney_v6);

// Mistral model evolution
MATCH (mistral7b:Model {name:"Mistral 7B"}), (mixtral:Model {name:"Mixtral 8x7B"})
CREATE (mistral7b)-[:EVOLVED_TO]->(mixtral);
MATCH (mixtral:Model {name:"Mixtral 8x7B"}), (mistral_large:Model {name:"Mistral Large"})
CREATE (mixtral)-[:EVOLVED_TO]->(mistral_large);
MATCH (mistral7b:Model {name:"Mistral 7B"}), (codestral:Model {name:"Codestral"})
CREATE (mistral7b)-[:SPECIALIZED_TO]->(codestral);
MATCH (mistral7b:Model {name:"Mistral 7B"}), (mathstral:Model {name:"Mathstral"})
CREATE (mistral7b)-[:SPECIALIZED_TO]->(mathstral);

// Aleph Alpha evolution
MATCH (luminous_base:Model {name:"Luminous Base"}), (luminous_extended:Model {name:"Luminous Extended"})
CREATE (luminous_base)-[:EVOLVED_TO]->(luminous_extended);
MATCH (luminous_extended:Model {name:"Luminous Extended"}), (luminous_supreme:Model {name:"Luminous Supreme"})
CREATE (luminous_extended)-[:EVOLVED_TO]->(luminous_supreme);

// Cohere model evolution
MATCH (command:Model {name:"Command"}), (command_r:Model {name:"Command R"})
CREATE (command)-[:EVOLVED_TO]->(command_r);
MATCH (command_r:Model {name:"Command R"}), (aya_vision:Model {name:"Aya Vision"})
CREATE (command_r)-[:EVOLVED_TO]->(aya_vision);

// Industry alliances and standards organizations
CREATE (forum:Org {name:"Frontier Model Forum", desc:"AI safety/standards: OpenAI, Google, Anthropic, Microsoft.", type:"Alliance"});
CREATE (partnership:Org {name:"Partnership on AI", desc:"Industry collaboration on AI best practices.", type:"Alliance"});
CREATE (mlcommons:Org {name:"MLCommons", desc:"ML benchmarking and standards organization.", type:"Standards"});

MATCH (forum:Org {name:"Frontier Model Forum"}), (openai:Org {name:"OpenAI"})
CREATE (forum)-[:FOUNDING_MEMBER]->(openai);
MATCH (forum:Org {name:"Frontier Model Forum"}), (anthropic:Org {name:"Anthropic"})
CREATE (forum)-[:FOUNDING_MEMBER]->(anthropic);
MATCH (forum:Org {name:"Frontier Model Forum"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (forum)-[:FOUNDING_MEMBER]->(googledeepmind);
MATCH (forum:Org {name:"Frontier Model Forum"}), (microsoft:Org {name:"Microsoft Research"})
CREATE (forum)-[:FOUNDING_MEMBER]->(microsoft);

MATCH (partnership:Org {name:"Partnership on AI"}), (openai:Org {name:"OpenAI"})
CREATE (partnership)-[:MEMBER]->(openai);
MATCH (partnership:Org {name:"Partnership on AI"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (partnership)-[:MEMBER]->(googledeepmind);
MATCH (partnership:Org {name:"Partnership on AI"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (partnership)-[:MEMBER]->(meta);
MATCH (partnership:Org {name:"Partnership on AI"}), (amazon:Org {name:"Amazon AI"})
CREATE (partnership)-[:MEMBER]->(amazon);
MATCH (partnership:Org {name:"Partnership on AI"}), (microsoft:Org {name:"Microsoft Research"})
CREATE (partnership)-[:MEMBER]->(microsoft);

// Government and regulatory relationships (expanded)
MATCH (nist:Org {name:"NIST AI Risk Management"}), (openai:Org {name:"OpenAI"})
CREATE (nist)-[:REGULATES]->(openai);
MATCH (nist:Org {name:"NIST AI Risk Management"}), (anthropic:Org {name:"Anthropic"})
CREATE (nist)-[:REGULATES]->(anthropic);
MATCH (nist:Org {name:"NIST AI Risk Management"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (nist)-[:REGULATES]->(googledeepmind);
MATCH (eu_ai_office:Org {name:"EU AI Office"}), (mistral:Org {name:"Mistral AI"})
CREATE (eu_ai_office)-[:REGULATES]->(mistral);
MATCH (eu_ai_office:Org {name:"EU AI Office"}), (aleph_alpha:Org {name:"Aleph Alpha"})
CREATE (eu_ai_office)-[:REGULATES]->(aleph_alpha);
MATCH (uk_ai_safety:Org {name:"UK AI Safety Institute"}), (deepmind:Org {name:"DeepMind"})
CREATE (uk_ai_safety)-[:COLLABORATES_WITH]->(deepmind);
MATCH (indiaai:Org {name:"IndiaAI Mission"}), (sarvam:Org {name:"Sarvam AI"})
CREATE (indiaai)-[:SUPPORTS]->(sarvam);
MATCH (indiaai:Org {name:"IndiaAI Mission"}), (corover:Org {name:"CoRover.ai"})
CREATE (indiaai)-[:SUPPORTS]->(corover);
MATCH (indiaai:Org {name:"IndiaAI Mission"}), (ganai:Org {name:"Gan AI Labs"})
CREATE (indiaai)-[:SUPPORTS]->(ganai);

// Technology licensing and dependencies
MATCH (openai:Org {name:"OpenAI"}), (microsoft_copilot:Org {name:"Microsoft Copilot"})
CREATE (openai)-[:LICENSES_TO]->(microsoft_copilot);
MATCH (meta:Org {name:"Meta AI (FAIR)"}), (mistral:Org {name:"Mistral AI"})
CREATE (meta)-[:INFLUENCED {via:"LLaMA architecture"}]->(mistral);
MATCH (googlebrain:Org {name:"Google Brain"}), (cohere:Org {name:"Cohere"})
CREATE (googlebrain)-[:INSPIRED {via:"Transformer architecture"}]->(cohere);
MATCH (openai:Org {name:"OpenAI"}), (characterai:Org {name:"Character.AI"})
CREATE (openai)-[:INSPIRED {via:"GPT models"}]->(characterai);
MATCH (anthropic:Org {name:"Anthropic"}), (chai:Org {name:"Chai AI"})
CREATE (anthropic)-[:INFLUENCED {via:"Constitutional AI"}]->(chai);

// Platform and ecosystem relationships
MATCH (huggingface:Org {name:"Hugging Face"}), (mistral:Org {name:"Mistral AI"})
CREATE (huggingface)-[:HOSTS_MODELS]->(mistral);
MATCH (huggingface:Org {name:"Hugging Face"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (huggingface)-[:HOSTS_MODELS]->(meta);
MATCH (huggingface:Org {name:"Hugging Face"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (huggingface)-[:HOSTS_MODELS]->(googledeepmind);
MATCH (huggingface:Org {name:"Hugging Face"}), (zhipu:Org {name:"Zhipu AI"})
CREATE (huggingface)-[:HOSTS_MODELS]->(zhipu);
MATCH (huggingface:Org {name:"Hugging Face"}), (deepseek:Org {name:"DeepSeek"})
CREATE (huggingface)-[:HOSTS_MODELS]->(deepseek);

// Cross-regional technology transfer
MATCH (openai:Org {name:"OpenAI"}), (zhipu:Org {name:"Zhipu AI"})
CREATE (openai)-[:INFLUENCED {via:"GPT architecture"}]->(zhipu);
MATCH (anthropic:Org {name:"Anthropic"}), (deepseek:Org {name:"DeepSeek"})
CREATE (anthropic)-[:INFLUENCED {via:"Constitutional AI principles"}]->(deepseek);
MATCH (googledeepmind:Org {name:"Google DeepMind"}), (baidu:Org {name:"Baidu"})
CREATE (googledeepmind)-[:INFLUENCED {via:"Transformer architecture"}]->(baidu);

// Business model relationships
MATCH (openai:Org {name:"OpenAI"}), (perplexity:Org {name:"Perplexity AI"})
CREATE (openai)-[:API_PROVIDER]->(perplexity);
MATCH (anthropic:Org {name:"Anthropic"}), (perplexity:Org {name:"Perplexity AI"})
CREATE (anthropic)-[:API_PROVIDER]->(perplexity);
MATCH (microsoft_copilot:Org {name:"Microsoft Copilot"}), (openai:Org {name:"OpenAI"})
CREATE (microsoft_copilot)-[:POWERED_BY]->(openai);
MATCH (sb_openai:Org {name:"SB OpenAI Japan"}), (openai:Org {name:"OpenAI"})
CREATE (sb_openai)-[:POWERED_BY]->(openai);

// Specialized AI ecosystems
CREATE (chinese_alliance:Org {name:"Chinese Model-Chip Ecosystem", desc:"Alliance of Chinese AI and chip companies.", type:"Alliance"});
MATCH (chinese_alliance:Org {name:"Chinese Model-Chip Ecosystem"}), (sensetime:Org {name:"SenseTime"})
CREATE (chinese_alliance)-[:INCLUDES]->(sensetime);
MATCH (chinese_alliance:Org {name:"Chinese Model-Chip Ecosystem"}), (megvii:Org {name:"Megvii"})
CREATE (chinese_alliance)-[:INCLUDES]->(megvii);
MATCH (chinese_alliance:Org {name:"Chinese Model-Chip Ecosystem"}), (yitu:Org {name:"YITU"})
CREATE (chinese_alliance)-[:INCLUDES]->(yitu);
MATCH (chinese_alliance:Org {name:"Chinese Model-Chip Ecosystem"}), (iflytek:Org {name:"iFlytek"})
CREATE (chinese_alliance)-[:INCLUDES]->(iflytek);
MATCH (chinese_alliance:Org {name:"Chinese Model-Chip Ecosystem"}), (deepseek:Org {name:"DeepSeek"})
CREATE (chinese_alliance)-[:INCLUDES]->(deepseek);
MATCH (chinese_alliance:Org {name:"Chinese Model-Chip Ecosystem"}), (cambricon:Org {name:"Cambricon"})
CREATE (chinese_alliance)-[:INCLUDES]->(cambricon);

// European AI sovereignty initiative
CREATE (eu_sovereign:Org {name:"European AI Sovereignty Initiative", desc:"EU initiative for independent AI capabilities.", type:"Alliance"});
MATCH (eu_sovereign:Org {name:"European AI Sovereignty Initiative"}), (mistral:Org {name:"Mistral AI"})
CREATE (eu_sovereign)-[:INCLUDES]->(mistral);
MATCH (eu_sovereign:Org {name:"European AI Sovereignty Initiative"}), (aleph_alpha:Org {name:"Aleph Alpha"})
CREATE (eu_sovereign)-[:INCLUDES]->(aleph_alpha);
MATCH (eu_sovereign:Org {name:"European AI Sovereignty Initiative"}), (huggingface:Org {name:"Hugging Face"})
CREATE (eu_sovereign)-[:INCLUDES]->(huggingface);

// Research publication and citation networks
MATCH (vaswani:Person {name:"Ashish Vaswani"}), (gomez:Person {name:"Aidan Gomez"})
CREATE (vaswani)-[:COAUTHORED_WITH {paper:"Attention Is All You Need"}]->(gomez);
MATCH (vaswani:Person {name:"Ashish Vaswani"}), (shazeer:Person {name:"Noam Shazeer"})
CREATE (vaswani)-[:COAUTHORED_WITH {paper:"Attention Is All You Need"}]->(shazeer);
MATCH (vaswani:Person {name:"Ashish Vaswani"}), (jones:Person {name:"Llion Jones"})
CREATE (vaswani)-[:COAUTHORED_WITH {paper:"Attention Is All You Need"}]->(jones);
MATCH (vaswani:Person {name:"Ashish Vaswani"}), (uszkoreit:Person {name:"Jakob Uszkoreit"})
CREATE (vaswani)-[:COAUTHORED_WITH {paper:"Attention Is All You Need"}]->(uszkoreit);
MATCH (vaswani:Person {name:"Ashish Vaswani"}), (parmar:Person {name:"Niki Parmar"})
CREATE (vaswani)-[:COAUTHORED_WITH {paper:"Attention Is All You Need"}]->(parmar);
MATCH (vaswani:Person {name:"Ashish Vaswani"}), (kaiser:Person {name:"Łukasz Kaiser"})
CREATE (vaswani)-[:COAUTHORED_WITH {paper:"Attention Is All You Need"}]->(kaiser);
MATCH (vaswani:Person {name:"Ashish Vaswani"}), (polosukhin:Person {name:"Illia Polosukhin"})
CREATE (vaswani)-[:COAUTHORED_WITH {paper:"Attention Is All You Need"}]->(polosukhin);

// Geographic AI hubs
CREATE (silicon_valley:Org {name:"Silicon Valley AI Hub", desc:"Concentration of AI companies in San Francisco Bay Area.", type:"Geographic"});
CREATE (london_ai:Org {name:"London AI Hub", desc:"UK AI research and startup ecosystem.", type:"Geographic"});
CREATE (beijing_ai:Org {name:"Beijing AI Hub", desc:"China's primary AI research and development center.", type:"Geographic"});
CREATE (toronto_ai:Org {name:"Toronto AI Hub", desc:"Canadian AI research cluster around University of Toronto.", type:"Geographic"});
CREATE (seoul_ai:Org {name:"Seoul AI Hub", desc:"South Korean AI development center.", type:"Geographic"});
CREATE (bangalore_ai:Org {name:"Bangalore AI Hub", desc:"India's Silicon Valley for AI development.", type:"Geographic"});

MATCH (silicon_valley:Org {name:"Silicon Valley AI Hub"}), (openai:Org {name:"OpenAI"})
CREATE (silicon_valley)-[:CONTAINS]->(openai);
MATCH (silicon_valley:Org {name:"Silicon Valley AI Hub"}), (anthropic:Org {name:"Anthropic"})
CREATE (silicon_valley)-[:CONTAINS]->(anthropic);
MATCH (silicon_valley:Org {name:"Silicon Valley AI Hub"}), (xai:Org {name:"xAI"})
CREATE (silicon_valley)-[:CONTAINS]->(xai);
MATCH (london_ai:Org {name:"London AI Hub"}), (deepmind:Org {name:"DeepMind"})
CREATE (london_ai)-[:CONTAINS]->(deepmind);
MATCH (london_ai:Org {name:"London AI Hub"}), (stability:Org {name:"Stability AI"})
CREATE (london_ai)-[:CONTAINS]->(stability);
MATCH (beijing_ai:Org {name:"Beijing AI Hub"}), (baidu:Org {name:"Baidu"})
CREATE (beijing_ai)-[:CONTAINS]->(baidu);
MATCH (beijing_ai:Org {name:"Beijing AI Hub"}), (zhipu:Org {name:"Zhipu AI"})
CREATE (beijing_ai)-[:CONTAINS]->(zhipu);
MATCH (toronto_ai:Org {name:"Toronto AI Hub"}), (cohere:Org {name:"Cohere"})
CREATE (toronto_ai)-[:CONTAINS]->(cohere);
MATCH (seoul_ai:Org {name:"Seoul AI Hub"}), (naver:Org {name:"Naver Corporation"})
CREATE (seoul_ai)-[:CONTAINS]->(naver);
MATCH (bangalore_ai:Org {name:"Bangalore AI Hub"}), (sarvam:Org {name:"Sarvam AI"})
CREATE (bangalore_ai)-[:CONTAINS]->(sarvam);

// ==== COMPREHENSIVE SAMPLE QUERIES ====
// 1. Find all companies founded by former OpenAI employees:
// MATCH (p:Person)-[:CHIEF_SCIENTIST|COFOUNDER|FOUNDING_MEMBER]->(openai:Org {name:"OpenAI"}),
//       (p)-[:COFOUNDER|FOUNDER|CEO]->(newOrg:Org)
// WHERE newOrg.name <> "OpenAI"
// RETURN p.name as Person, collect(DISTINCT newOrg.name) as NewCompanies;

// 2. Find the complete GPT model evolution chain:
// MATCH path = (gpt1:Model {name:"GPT-1"})-[:EVOLVED_TO*]->(latest:Model)
// WHERE NOT EXISTS((latest)-[:EVOLVED_TO]->())
// RETURN path;

// 3. Find all Transformer paper co-authors and their current companies after Google:
// MATCH (p:Person)-[:COFOUNDER|CEO|RESEARCH_SCIENTIST]->(org:Org)
// WHERE p.name IN ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", 
//                  "Llion Jones", "Aidan Gomez", "Łukasz Kaiser", "Illia Polosukhin"]
//   AND org.name <> "Google Brain"
// RETURN p.name, collect(org.name) as CurrentCompanies, org.type;

// 4. Find complete investment network (multi-hop):
// MATCH path = (investor:Org)-[:INVESTED_IN*1..2]->(company:Org)-[:INVESTED_IN*1..2]->(final:Org)
// WHERE investor.type = "Investment" OR investor.type = "Big Tech"
// RETURN path LIMIT 10;

// 5. Find academic mentorship chains leading to company founders:
// MATCH path = (professor:Person)-[:MENTORED*1..3]->(student:Person)-[:FOUNDER|COFOUNDER|CEO]->(company:Org)
// WHERE company.type = "Startup"
// RETURN path;

// 6. Find all AI models by generation and their competition:
// MATCH (model1:Model)-[:COMPETES_WITH]-(model2:Model)
// WHERE model1.year >= 2023 AND model2.year >= 2023
// RETURN model1.name, model1.generation, model2.name, model2.generation, 
//        model1.year, model2.year
// ORDER BY model1.year DESC;

// 7. Find cross-regional AI ecosystem connections:
// MATCH (western:Org)-[r:INFLUENCED|INSPIRED|API_PROVIDER]-(eastern:Org)
// WHERE (western.location CONTAINS "CA" OR western.location CONTAINS "WA" OR western.location CONTAINS "UK")
//   AND (eastern.location CONTAINS "China" OR eastern.location CONTAINS "Japan" OR eastern.location CONTAINS "Korea")
// RETURN western.name, type(r), eastern.name, western.location, eastern.location;

// 8. Find hardware dependency network across global AI companies:
// MATCH (hardware:Org)-[:PROVIDES_HARDWARE|PROVIDES_INFERENCE|PROVIDES_TRAINING]->(ai:Org)
// WHERE hardware.type = "Hardware"
// RETURN hardware.name, hardware.location, collect(ai.name) as AICompanies, 
//        collect(ai.location) as AILocations;

// 9. Find all government-supported AI initiatives by country:
// MATCH (gov:Org)-[:SUPPORTS|REGULATES|COLLABORATES_WITH]->(ai:Org)
// WHERE gov.type = "Government"
// RETURN gov.name, gov.location, collect(ai.name) as SupportedCompanies;

// 10. Find model family trees (complete evolution paths):
// MATCH path = (root:Model)-[:EVOLVED_TO|SPECIALIZED_TO*]->(leaf:Model)
// WHERE NOT EXISTS(()-[:EVOLVED_TO|SPECIALIZED_TO]->(root))
//   AND NOT EXISTS((leaf)-[:EVOLVED_TO|SPECIALIZED_TO]->())
// RETURN path;

// 11. Find talent flow between major AI labs:
// MATCH (person:Person)-[r1:RESEARCH_SCIENTIST|COFOUNDER]->(org1:Org),
//       (person)-[r2:RESEARCH_SCIENTIST|COFOUNDER|CEO]->(org2:Org)
// WHERE org1.name <> org2.name 
//   AND (org1.type = "Big Tech" OR org1.type = "Startup")
//   AND (org2.type = "Big Tech" OR org2.type = "Startup")
//   AND r1.left IS NOT NULL
// RETURN person.name, org1.name as From, org2.name as To, r1.left as LeftYear;

// 12. Find multilingual and regional language model ecosystems:
// MATCH (org:Org)-[:RELEASED]->(model:Model)
// WHERE model.desc CONTAINS "Chinese" OR model.desc CONTAINS "Korean" 
//    OR model.desc CONTAINS "Japanese" OR model.desc CONTAINS "Indian"
//    OR model.desc CONTAINS "Hindi" OR model.desc CONTAINS "multilingual"
// RETURN org.name, org.location, model.name, model.desc;

// 13. Find complete competitive landscape by model generation:
// MATCH (org1:Org)-[:RELEASED]->(model1:Model)-[:COMPETES_WITH]-(model2:Model)<-[:RELEASED]-(org2:Org)
// WHERE model1.year = 2024 OR model2.year = 2024
// RETURN org1.name as Company1, model1.name as Model1, 
//        org2.name as Company2, model2.name as Model2,
//        model1.generation, model2.generation;

// 14. Find research partnerships leading to commercial applications:
// MATCH (university:Org)-[:RESEARCH_PARTNERSHIP]->(company:Org)-[:RELEASED]->(model:Model)
// WHERE university.type = "Academic" AND company.type IN ["Big Tech", "Startup"]
// RETURN university.name, company.name, collect(model.name) as Models;

// 15. Find complete supply chain from hardware to end-user applications:
// MATCH path = (hardware:Org)-[:PROVIDES_HARDWARE]->(ai:Org)-[:RELEASED]->(model:Model)
// WHERE hardware.type = "Hardware" AND model.type = "Interface"
// RETURN path;rok2:Model {name:"Grok-2"})
CREATE (gpt4)-[:COMPETES_WITH]->(grok2);
MATCH (dalle3:Model {name:"DALL-E 3"}), (midjourney_v6:Model {name:"Midjourney v6"})
CREATE (dalle3)-[:COMPETES_WITH]->(midjourney_v6);
MATCH (dalle3:Model {name:"DALL-E 3"}), (stable_diffusion_xl:Model {name:"Stable Diffusion XL"})
CREATE (dalle3)-[:COMPETES_WITH]->(stable_diffusion_xl);
MATCH (chatgpt:Model {name:"ChatGPT"}), (perplexity_model:Model {name:"Perplexity LLM"})
CREATE (chatgpt)-[:COMPETES_WITH]->(perplexity_model);

// Regional competition
MATCH (gpt4:Model {name:"GPT-4"}), (deepseek_v3:Model {name:"DeepSeek-V3"})
CREATE (gpt4)-[:COMPETES_WITH]->(deepseek_v3);
MATCH (claude35_sonnet:Model {name:"Claude 3.5 Sonnet"}), (qwen2:Model {name:"Qwen 2"})
CREATE (claude35_sonnet)-[:COMPETES_WITH]->(qwen2);
MATCH (gemini_pro:Model {name:"Gemini Pro"}), (glm45:Model {name:"GLM-4.5"})
CREATE (gemini_pro)-[:COMPETES_WITH]->(glm45);
MATCH (gpt4:Model {name:"GPT-4"}), (hyperclova:Model {name:"HyperClova X"})
CREATE (gpt4)-[:COMPETES_WITH]->(hyperclova);
MATCH (chatglm:Model {name:"ChatGLM"}), (sarvam_m:Model {name:"Sarvam M"})
CREATE (chatglm)-[:COMPETES_WITH]->(sarvam_m);

// Research partnerships and collaborations (expanded)
MATCH (stanford:Org {name:"Stanford AI Lab"}), (openai:Org {name:"OpenAI"})
CREATE (stanford)-[:RESEARCH_PARTNERSHIP]->(openai);
MATCH (mit:Org {name:"MIT AI Lab"}), (googledeepmind:Org {name:"Google DeepMind"})
CREATE (mit)-[:RESEARCH_PARTNERSHIP]->(googledeepmind);
MATCH (toronto:Org {name:"University of Toronto"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (toronto)-[:RESEARCH_PARTNERSHIP]->(meta);
MATCH (cmu:Org {name:"Carnegie Mellon AI"}), (openai:Org {name:"OpenAI"})
CREATE (cmu)-[:RESEARCH_PARTNERSHIP]->(openai);
MATCH (oxford:Org {name:"Oxford AI"}), (deepmind:Org {name:"DeepMind"})
CREATE (oxford)-[:RESEARCH_PARTNERSHIP]->(deepmind);
MATCH (montreal:Org {name:"MILA Montreal"}), (meta:Org {name:"Meta AI (FAIR)"})
CREATE (montreal)-[:RESEARCH_PARTNERSHIP]->(meta);
MATCH (montreal:Org {name:"MILA Montreal"}), (cohere:Org {name:"Cohere"})
CREATE (montreal)-[:RESEARCH_PARTNERSHIP]->(cohere);
MATCH (tsinghua:Org {name:"Tsinghua University"}), (zhipu:Org {name:"Zhipu AI"})
CREATE (tsinghua)-[:RESEARCH_PARTNERSHIP]->(zhipu);
MATCH (kaist:Org {name:"KAIST"}), (naver:Org {name:"Naver Corporation"})
CREATE (kaist)-[:RESEARCH_PARTNERSHIP]->(naver);

// Model evolution chains (detailed)
MATCH (gpt1:Model {name:"GPT-1"}), (gpt2:Model {name:"GPT-2"})
CREATE (gpt1)-[:EVOLVED_TO]->(gpt2);
MATCH (gpt2:Model {name:"GPT-2"}), (gpt3:Model {name:"GPT-3"})
CREATE (gpt2)-[:EVOLVED_TO]->(gpt3);
MATCH (gpt3:Model {name:"GPT-3"}), (gpt35:Model {name:"GPT-3.5"})
CREATE (gpt3)-[:EVOLVED_TO]->(gpt35);
MATCH (gpt35:Model {name:"GPT-3.5"}), (gpt4:Model {name:"GPT-4"})
CREATE (gpt35)-[:EVOLVED_TO]->(gpt4);
MATCH (gpt4:Model {name:"GPT-4"}), (g
