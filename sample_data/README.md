NAB Data Corpus
---

Dataset from: https://github.com/lin-shuyu/VAE-LSTM-for-anomaly-detection/tree/master/datasets/NAB-known-anomaly


### Real data 
- realKnownCause/

	This is data for which we know the anomaly causes; no hand labeling.
	
	- ambient_temperature_system_failure.csv: The ambient temperature in an office
	setting.
	- cpu_utilization_asg_misconfiguration.csv: From Amazon Web Services (AWS)
	monitoring CPU usage – i.e. average CPU usage across a given cluster. When
	usage is high, AWS spins up a new machine, and uses fewer machines when usage
	is low.
	- ec2_request_latency_system_failure.csv: CPU usage data from a server in
	Amazon's East Coast datacenter. The dataset ends with complete system failure
	resulting from a documented failure of AWS API servers. There's an interesting
	story behind this data in the [Numenta
	blog](http://numenta.com/blog/anomaly-of-the-week.html).
	- machine_temperature_system_failure.csv: Temperature sensor data of an
	internal component of a large, industrial mahcine. The first anomaly is a
	planned shutdown of the machine. The second anomaly is difficult to detect and
	directly led to the third anomaly, a catastrophic failure of the machine.
	- nyc_taxi.csv: Number of NYC taxi passengers, where the five anomalies occur
	during the NYC marathon, Thanksgiving, Christmas, New Years day, and a snow
	storm. The raw data is from the [NYC Taxi and Limousine Commission](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml).
	The data file included here consists of aggregating the total number of
	taxi passengers into 30 minute buckets.