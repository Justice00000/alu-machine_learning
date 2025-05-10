<<<<<<< HEAD
-- lists records with score >= 10 and order by score(top first)
SELECT score, name FROM second_table
WHERE score >= 10 
ORDER BY score DESC;

=======
-- Lists all records with score >= 10 in second_table in MySQL server
-- results should display both score and name, in descending order by score
-- database name will be passed as an argument of the mysql command
SELECT `score`, `name` FROM second_table WHERE `score` >= 10 ORDER BY `score` DESC;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
