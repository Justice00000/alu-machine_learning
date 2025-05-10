<<<<<<< HEAD
-- Create table
CREATE TABLE
    IF NOT EXISTS users (
        id INT NOT Null AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) NOT NULL UNIQUE,
        name VARCHAR(255)
    );
=======
-- Creates a table users with id, email, and name
-- id: integer, never null, auto increment, and primary key
-- email: string (255 chars), never null, unique
-- name: string (255 chars)
-- database name will be passed as an argument of the mysql command
CREATE TABLE IF NOT EXISTS users (
       id INT NOT NULL AUTO_INCREMENT,
       email VARCHAR(255) UNIQUE NOT NULL,
       name VARCHAR (255),
       PRIMARY KEY (id)
       );
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
