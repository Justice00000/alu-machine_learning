<<<<<<< HEAD
-- display average temprature
SELECT city, AVG(value) as avg_temp
FROM temperatures 
GROUP BY city 
ORDER BY avg_temp DESC;
=======
-- Displays the average temperature by city ordered by temperature (desc)
-- uses hbtn_0c_0 database
-- database name will be passed as an argument of the mysql command
SELECT city, AVG(value) AS avg_temp FROM temperatures GROUP BY city ORDER BY avg_temp DESC;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
