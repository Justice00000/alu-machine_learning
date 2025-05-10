<<<<<<< HEAD
-- Agriggation
SELECT title, SUM(tv_show_ratings.rate) AS rating
FROM tv_shows
JOIN tv_show_ratings ON tv_shows.id = tv_show_ratings.show_id
GROUP BY title
ORDER BY rating DESC;
=======
-- Lists all shows from hbtn_0d_tvshows_rate by their rating
-- uses hbtn_0d_tvshows_rate database
-- database name will be passed as an argument of the mysql command
SELECT t.title AS title, SUM(r.rate) AS rating
       FROM tv_shows AS t
       LEFT JOIN tv_show_ratings AS r
       ON t.id = r.show_id
       GROUP BY t.title
       ORDER BY rating DESC;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
