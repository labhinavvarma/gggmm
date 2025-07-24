MATCH (eda:Group {name: "EDA"})-[rel]-(eai:Group {name: "EAI"})
RETURN eda, rel, eai
