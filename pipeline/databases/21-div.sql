<<<<<<< HEAD
-- Content: SQL to create a function that safely divides two numbers
DELIMITER $$
CREATE FUNCTION SafeDiv(a INT, b INT) RETURNS FLOAT

BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN a / b;
    END IF;
END $$
=======
-- Creates a function SafeDiv that divides (and returns) the first by the second number
--    or returns 0 if the second number is 0
-- Function SafeDiv takes 2 arguments:
--    a, INT
--    b, INT
-- Returns
--    a / b
--    0, if b == 0
DELIMITER //

CREATE FUNCTION SafeDiv (a INT, b INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
        IF b = 0 THEN
           RETURN 0;
        ELSE
                RETURN (a / b);
        END IF;
END //
DELIMITER ;
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
