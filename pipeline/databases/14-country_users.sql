<<<<<<< HEAD
-- create table 
CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US'
);
=======
-- Creates a table users with id, email, name, and country
-- id: integer, never null, auto increment, and primary key
-- email: string (255 chars), never null, unique
-- name: string (255 chars)
-- country: enumeration of countries: US, CO, TN, never null, default=first (US)
-- database name will be passed as an argument of the mysql command
CREATE TABLE IF NOT EXISTS users (
       id INT NOT NULL AUTO_INCREMENT,
       email VARCHAR(255) UNIQUE NOT NULL,
       name VARCHAR (255),
       country ENUM('US', 'CO', 'TN') DEFAULT 'US' NOT NULL,
       PRIMARY KEY (id)
       );
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
