Here are Neo4j Cypher commands to check CEO relationships:
👑 CEO Relationship Commands:
1. Find All CEO Relationships
cypherMATCH (ceo:Person {name: "Sarah Johnson"})-[r]-(other:Person) 
RETURN ceo.name, type(r), other.name, other.title
2. Find Who Reports TO the CEO
cypherMATCH (ceo:Person {name: "Sarah Johnson"})<-[:REPORTS_TO]-(employee:Person) 
RETURN ceo.name as CEO, employee.name as DirectReport, employee.title
3. Find CEO's Entire Team (All Levels)
cypherMATCH (ceo:Person {name: "Sarah Johnson"})<-[:REPORTS_TO*1..]-(employee:Person) 
RETURN ceo.name as CEO, employee.name as TeamMember, employee.title, length(path) as Level
ORDER BY Level, employee.name
4. Count CEO's Direct Reports
cypherMATCH (ceo:Person {name: "Sarah Johnson"})<-[:REPORTS_TO]-(direct:Person) 
RETURN ceo.name as CEO, count(direct) as DirectReports
5. Show CEO's Management Structure
cypherMATCH path = (ceo:Person {name: "Sarah Johnson"})<-[:REPORTS_TO*1..3]-(employee:Person) 
RETURN ceo.name as CEO, employee.name as Employee, employee.title, length(path) as ReportingLevel
ORDER BY ReportingLevel, employee.name
6. Find All Relationship Types for CEO
cypherMATCH (ceo:Person {name: "Sarah Johnson"})-[r]-() 
RETURN DISTINCT type(r) as RelationshipTypes
7. CEO's Complete Network
cypherMATCH (ceo:Person {name: "Sarah Johnson"})-[r]-(connected) 
RETURN ceo.name as CEO, type(r) as Relationship, 
       connected.name as ConnectedPerson, labels(connected) as NodeType
8. Visualization Query (Shows Full Hierarchy)
cypherMATCH path = (ceo:Person {name: "Sarah Johnson"})<-[:REPORTS_TO*]-(employee:Person) 
RETURN path
🔍 Advanced Relationship Queries:
9. Find Management Chain Lengths
cypherMATCH (ceo:Person {name: "Sarah Johnson"})<-[:REPORTS_TO*]-(emp:Person) 
WITH length(path) as levels, count(emp) as people
RETURN levels, people 
ORDER BY levels
10. Find CEO's Peers (Same Level)
cypherMATCH (ceo:Person {name: "Sarah Johnson"})
MATCH (peer:Person) 
WHERE peer.title CONTAINS "CEO" OR peer.title CONTAINS "President"
RETURN peer.name, peer.title
11. Two-Way Relationships Check
cypherMATCH (ceo:Person {name: "Sarah Johnson"})-[r1]->(other:Person)-[r2]->(ceo) 
RETURN ceo.name, type(r1), other.name, type(r2)
