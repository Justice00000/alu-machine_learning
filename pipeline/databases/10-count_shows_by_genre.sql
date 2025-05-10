<<<<<<< HEAD
-- count
SELECT name AS genre, COUNT(tv_show_genres.show_id) AS number_of_shows
FROM tv_genres
JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
GROUP BY genre
ORDER BY number_of_shows DESC
=======
-- Lists all genres and displays the number of shows linked to each
-- uses hbtn_0d_tvshows database
-- database name will be passed as an argument of the mysql command
SELECT g.name AS genre, COUNT(t.show_id) AS number_of_shows
       FROM tv_genres AS g
       JOIN tv_show_genres AS t
       ON g.id = t.genre_id
       GROUP BY g.id
       ORDER BY number_of_shows DESC;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
