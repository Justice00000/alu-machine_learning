<<<<<<< HEAD
-- Create a stored procedure to compute the average score
-- for a user and update the average_score column in the users table.
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser (
    IN user_id_param INT
)
BEGIN
    DECLARE user_avg_score FLOAT;
    
    -- Calculate average score for the user
    SELECT AVG(score) INTO user_avg_score
    FROM corrections
    WHERE user_id = user_id_param;
    
    -- Update the average_score column in the users table
    UPDATE users
    SET average_score = user_avg_score
    WHERE id = user_id_param;
END //

=======
-- Creates a stored procedure ComputeAverageScoreForUser that computes and stores the average score for a student
-- Procedure AddBonus takes 1 input:
--    user_id, a users.id value, can assume user_id is linked to existing users
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id_new INTEGER)
BEGIN
	UPDATE users SET average_score=(
	SELECT AVG(score) FROM corrections WHERE user_id=user_id_new)
	WHERE id=user_id_new;
END; //
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
DELIMITER ;
