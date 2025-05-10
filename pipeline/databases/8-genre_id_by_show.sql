<<<<<<< HEAD
-- join
SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
ORDER BY title ASC, tv_show_genres.genre_id ASC
=======
-- Lists all shows in hbtn_0d_tvshows that have at least one genre linked
-- uses hbtn_0d_tvshows database
-- database name will be passed as an argument of the mysql command
SELECT s.`title`, g.`genre_id` FROM `tv_shows` AS s INNER JOIN `tv_show_genres` AS g ON s.`id` = g.`show_id` ORDER BY s.`title`, g.`genre_id` ASC;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
