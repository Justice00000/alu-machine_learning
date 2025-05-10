<<<<<<< HEAD
-- computes the score average
SELECT AVG(score) as average FROM second_table;
=======
-- Computes the average of all records in second_table in MySQL server
-- resulting column name should be 'average'
-- database name will be passed as an argument of the mysql command
SELECT AVG(`score`) 'average' FROM `second_table`;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
