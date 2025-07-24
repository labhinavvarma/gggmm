MATCH (eda:Subsidiary {name: "EDA"})-[rel]-(eai:Subsidiary {name: "EAI"})
RETURN type(rel) AS relationship_type, properties(rel) AS relationship_properties
