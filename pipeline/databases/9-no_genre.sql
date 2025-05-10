<<<<<<< HEAD
-- join
SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
LEFT JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
WHERE tv_show_genres.genre_id IS NULL
ORDER BY title ASC, tv_show_genres.genre_id ASC
=======
-- Lists all shows in hbtn_0d_tvshows without a genre linked
-- uses hbtn_0d_tvshows database
-- database name will be passed as an argument of the mysql command
SELECT s.`title`, g.`genre_id` FROM `tv_shows` AS s LEFT JOIN `tv_show_genres` AS g ON s.`id` = g.`show_id` WHERE g.`show_id` IS NULL ORDER BY s.`title` ASC;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
