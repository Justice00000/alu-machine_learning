<<<<<<< HEAD
-- find the lifespan of glam rock bands
SELECT band_name, IFNULL(split, 2020) - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;
=======
-- Lists all bands with Glam rock as their main style, ranked by longevity
-- utilizes the metal_bands table
SELECT band_name, IF(split IS NULL, (2020 - formed), (split - formed)) AS lifespan
       FROM metal_bands
       WHERE `style` LIKE '%Glam rock%'
       ORDER BY lifespan DESC;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
