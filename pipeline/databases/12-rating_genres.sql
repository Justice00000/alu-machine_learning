<<<<<<< HEAD
-- nested join
SELECT tv_genres.name, SUM(tv_show_ratings.rate) AS rating
FROM tv_genres
INNER JOIN tv_show_genres
ON tv_genres.id = tv_show_genres.genre_id
INNER JOIN tv_show_ratings
ON tv_show_genres.show_id = tv_show_ratings.show_id
GROUP BY name
ORDER BY rating DESC;
=======
-- Lists all genres in the database by their rating
-- uses hbtn_0d_tvshows_rate database
-- database name will be passed as an argument of the mysql command
SELECT g.name AS name, SUM(r.rate) AS rating
       FROM tv_genres AS g
       LEFT JOIN tv_show_genres AS t
       ON g.id = t.genre_id
       LEFT JOIN tv_show_ratings AS r
       ON t.show_id = r.show_id
       GROUP BY g.name
       ORDER BY rating DESC;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
