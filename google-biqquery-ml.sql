-- CREATE TABLE `automl-446814.data_driven_ml.ccpp_cleaned` 
-- AS
-- SELECT * 
-- FROM `automl-446814.data_driven_ml.ccpp_raw`
-- WHERE
-- Temp BETWEEN 1.81 AND 37.11 AND
-- Ambient_Pressure BETWEEN 992.89 AND 1033.30 AND
-- Relative_Humidity BETWEEN 25.56 AND 100.16 AND
-- Exhaust_Vacuum BETWEEN 25.36 AND 81.56 AND 
-- Energy_Production BETWEEN 420.26 AND 495.76;

CREATE OR REPLACE MODEL `data_driven_ml.energy_production`
OPTIONS (model_type='linear_reg', input_label_cols=['Energy_Production']) AS
SELECT Temp,
Ambient_Pressure,
Relative_Humidity,
Exhaust_Vacuum,
Energy_Production
FROM
`automl-446814.data_driven_ml.ccpp_cleaned`;

SELECT * FROM ML.EVALUATE(MODEL `data_driven_ml.energy_production`);

SELECT * FROM
ML.PREDICT(MODEL `data_driven_ml.energy_production`,
(SELECT 
  27.45 AS Temp,
  1001.23 AS Ambient_Pressure,
  84 AS Relative_Humidity,
  65.12 AS Exhaust_Vacuum));

CREATE OR REPLACE MODEL `data_driven_ml.energy_production`
OPTIONS (
  model_type='linear_reg', 
  input_label_cols=['Energy_Production'],
  enable_global_explain=TRUE
) AS
SELECT Temp,
Ambient_Pressure,
Relative_Humidity,
Exhaust_Vacuum,
Energy_Production
FROM
`automl-446814.data_driven_ml.ccpp_cleaned`;

SELECT * FROM
ML.GLOBAL_EXPLAIN(MODEL `data_driven_ml.energy_production`);

SELECT * FROM
ML.EXPLAIN_PREDICT(
  MODEL `data_driven_ml.energy_production`,
  (
    SELECT 
      Temp,
      Ambient_Pressure,
      Relative_Humidity,
      Exhaust_Vacuum
    FROM 
      `automl-446814.data_driven_ml.ccpp_cleaned`
  ),
  STRUCT(3 AS top_k_features)
);

CREATE OR REPLACE MODEL `data_driven_ml.energy_production_nn`
OPTIONS (
  model_type='dnn_regressor',
  hidden_units=[32, 16, 8],
  input_label_cols=['Energy_Production']
) AS 
SELECT
  Temp,
  Ambient_Pressure,
  Relative_Humidity,
  Exhaust_Vacuum,
  Energy_Production
FROM
  `automl-446814.data_driven_ml.ccpp_cleaned`;

