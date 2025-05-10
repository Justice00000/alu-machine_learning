<<<<<<< HEAD
-- sum of fans
SELECT origin, SUM(fans) nb_fans
FROM metal_bands
GROUP BY origin
ORDER BY nb_fans DESC;
=======
-- Ranks country origins of bands, ordered by the number of non-unique fans
-- utilizes the metal_bands table
SELECT origin, SUM(fans) AS nb_fans
       FROM metal_bands
       GROUP BY origin
       ORDER BY nb_fans DESC;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
