MATCH (eda:Subsidiary {name: "EDA"})-[rel]->(eai:Subsidiary {name: "EAI"})
RETURN eda, rel, eai

