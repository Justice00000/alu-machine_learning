<<<<<<< HEAD
-- Procedure to add a bonus to a user for a project
DELIMITER $$

CREATE PROCEDURE AddBonus(
    IN p_user_id INT,
    IN p_project_name VARCHAR(255),
    IN p_score INT
)
BEGIN
    DECLARE project_id INT;

    -- Check if project exists
    SELECT id INTO project_id FROM projects WHERE name = p_project_name;

    -- If project doesn't exist, create it
    IF project_id IS NULL THEN
        INSERT INTO projects (name) VALUES (p_project_name);
        SET project_id = LAST_INSERT_ID();
    END IF;

    -- Add correction
    INSERT INTO corrections (user_id, project_id, score) VALUES (p_user_id, project_id, p_score);
END $$

=======
-- Creates a stored procedure AddBonus that adds a new correction for a student
-- Procedure AddBonus takes 3 inputs (in this order)
--    user_id, a users.id value, can assume user_id is linked to existing users
--    project_name, a new or already exists projects - if not projects.name found, create it
--    score, the score value for correction
DELIMITER //

CREATE PROCEDURE AddBonus (IN user_id INTEGER, IN project_name VARCHAR(255), IN score INTEGER)
BEGIN
	IF NOT EXISTS (SELECT name FROM projects WHERE name=project_name) THEN
	   INSERT INTO projects(name)
	   VALUES (project_name);
	END IF;
	INSERT INTO corrections(user_id, project_id, score)
	VALUES(user_id, (SELECT id FROM projects WHERE name=project_name), score);
END; //
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
DELIMITER ;
