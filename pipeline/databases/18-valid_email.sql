<<<<<<< HEAD
-- Validate email
DELIMITER $$
CREATE TRIGGER reset_valid_email
BEFORE UPDATE ON users
FOR EACH ROW
=======
-- Creates a trigger that decreases the quantity of an item after adding a new order
-- quantity in the table `items` can be negative
DROP TRIGGER IF EXISTS reset_validation;

DELIMITER $$
CREATE TRIGGER reset_validation
       BEFORE UPDATE
       ON `users` FOR EACH ROW
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
BEGIN
	IF STRCMP(old.email, new.email) <> 0 THEN
	   SET new.valid_email = 0;
	END IF;
<<<<<<< HEAD
END $$
=======
END $$

DELIMITER ;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
