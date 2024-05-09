# Predictive-Maintenance-for-Wind-Turbines
Abstract:
A lot of data such as speed, temperature, status, power is continuously gathered by windmill sensors. This data can be analyzed using ML techniques to find trends that could indicate component problems. This increases the lifespan of windmills by reducing unscheduled stops.

Methodology:
•	Data is acquired from 5 Skystream power turbines. Apache Kafka helps gather this data from the turbines.
•	Next this data is integrated into PySpark where Kafka Messages are retrieved and meticulously organized into distinct data frames for each turbine.
•	Following this integration, the data is pre-processed eliminating irregularities.
•	Previous years’ wind turbine data is used to train the a LSTM model as lstm can process sequential data.
•	Then this model is used for live predictions in a UI which shows the user status code based on the current predictions.
