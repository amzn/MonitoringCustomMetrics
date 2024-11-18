# MonitoringCustomMetrics

`MonitoringCustomMetrics` is a code package that simplifies the creation of metrics to use for monitoring Machine Learning files. We follow 
the formats and standards defined by [Amazon SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html).
It can be executed locally by using Docker, or it can be used within a SageMaker Processing Job.

## What does it do?

This tool helps you monitor the quality of ML models with metrics that are not present in Amazon SageMaker Model Monitor. We follow 
SageMaker standards for metric output:

- [Statistics file](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-interpreting-statistics.html): raw statistics calculated
per column/feature. They are calculated for the baseline and also for the current input being analyzed.
- [Constraints file](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-constraints.html): these are the constraints
that a dataset must satisfy. The constraints are used to determine if the dataset has violations when running an evaluation job.
- [Constraint violations file](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-interpreting-violations.html): generated as
the output of a monitor execution. It contains the list of constraints evaluated (using a provided constraints file) against the dataset
being analyzed.

To avoid filename conflicts with SageMaker Monitor output, our files are renamed to:

- community_statistics.json
- community_constraints.json
- community_constraint_violations.json

## Operation modes

The package has two operation modes:

- Suggest baseline: as the name implies, this operation mode will suggest a baseline that you can later use for evaluating statistics.
It will generate "statistics" and "constraints" files. You will need to provide the input file(s) to be evaluated.
In case of Model Quality metrics, a "parameters.json" file is also needed in order to specify the metrics to evaluate and any additional
required parameter.
- Run monitor: it evaluates the input file(s) using the constraints provided. It will generate a "constraint_violations" file. 

It can perform both Data Quality and Model Quality analyses. The input can be a single file, or it can be split into multiple files.

### Data Quality

Data Quality analysis will evaluate all the existing metrics against all the columns. Based on the inferred column type, the package will 
run either "numerical" or "string" metrics on a given column.

### Model Quality

Model Quality analysis will only evaluate metrics specified in the configuration file provided.


## Known limitations

- The code runs on a single machine. If running on a SageMaker Processing Job, it will be limited to the capacity of a single instance.
- Pandas loads data in memory. Choose a host that can handle the amount data you need to process.
- `MonitoringCustomMetrics` expects the input file(s) to be in CSV format (comma-separated files).

# Running the package locally

`MonitoringCustomMetrics` can be executed locally. You will need to install Docker CLI, set the needed parameters in the Dockerfile, and provide
the required input file(s).

## Prerequisites

Before running locally, you will need to install Docker CLI:

https://docs.docker.com/get-started/get-docker/

https://docs.docker.com/reference/cli/docker/

## Environment variables

The package uses the following variables:

- analysis_type: specifies the type of analysis to do.
  - Possible values: 
    - DATA_QUALITY.
    - MODEL_QUALITY.
  - Required: Yes.
- baseline_statistics: specifies the container path to the baseline statistics file.
  - Required: only if you want to evaluate statistics. Not required when suggesting baseline.
- baseline_constraints: specifies the container path to the baseline constraints file.
 - Required: only if you want to evaluate statistics. Not required when suggesting baseline.

Model Quality specific environment variables:

- config_path: specifies the container path to the configuration file.
  - Required: only for Model Quality metrics. You need to specify the metric(s) to use, as well as any required parameter.
- problem_type: problem type for the analysis.
  - Required: Yes.
  - Possible values:
    - BinaryClassification
    - Regression
    - MulticlassClassification

- To specify that this is a Data Quality analysis:
  ```
  ENV analysis_type=DATA_QUALITY
  ```

- To specify that this is a Model Quality analysis:
  ```
  ENV analysis_type=MODEL_QUALITY
  ```

- If you want to evaluate statistics, you also need to provide the location of statistics and constraints files inside the container.
If these files are not provided, the package will suggest a baseline instead.
  ```
  ENV baseline_statistics=/opt/ml/processing/baseline/statistics/community_statistics.json
  ENV baseline_constraints=/opt/ml/processing/baseline/constraints/community_constraints.json
  ```

### Model Quality specific environment variables

For Model Quality, 'config_path' is also required:

`config_path` specifies the location of the "parameters" file within the container.
```
ENV config_path=/opt/ml/processing/input/parameters
```

Depending on the metrics to use, these variables might be needed also:
```
ENV problem_type=<problem type>
ENV ground_truth_attribute=<ground truth attribute column>
ENV inference_attribute=<inference attribute column>
```

#### Model Quality parameters file

Only the metrics specified in the "parameters" file will be evaluated in a Model Quality job. The parameters file is structured as a map,
with the top-level representing the metric names to use. For example:

```
{
  "prc_auc": {
    "threshold_override": 55
  }
}
```
would mean that the job will only evaluate the "prc_auc" metric, and it will pass parameter "threshold_override" with value "55".  


## Providing input files

The container also needs certain files to do the analysis. You can put your files in the "local_resources" directory. Once the files are
present, you need to add the following statements to the Dockerfile to have them copied over to the container:

- Copy the input data file. Input data can be split across multiple files if needed:
  ```
  COPY local_resources/data_quality/input.csv /opt/ml/processing/input/data
  ```

- Copy statistics and constraints files, if needed: 
  ```
  COPY local_resources/model_quality/community_constraints.json /opt/ml/processing/baseline/constraints
  COPY local_resources/model_quality/community_statistics.json /opt/ml/processing/baseline/statistics
  ```
  
- Copy "parameters" file, if needed (only needed for Model Monitoring metrics):
  ```
  COPY local_resources/model_quality/binary_classification/custom_metric/parameters.json /opt/ml/processing/input/parameters
  ```

## Running the container locally

Add the required parameters to the Dockerfile in the section specified. It should look something like:

```
##### Parameters for running locally should be put here: #####################################
ENV analysis_type=DATA_QUALITY
ENV baseline_statistics=/opt/ml/processing/baseline/statistics/community_statistics.json
ENV baseline_constraints=/opt/ml/processing/baseline/constraints/community_constraints.json
COPY local_resources/data_quality/input.csv /opt/ml/processing/input/data
COPY local_resources/data_quality/community_constraints.json /opt/ml/processing/baseline/constraints
COPY local_resources/data_quality/community_statistics.json /opt/ml/processing/baseline/statistics
##### End of Parameters for running locally ###########################################################################################
```

You can now execute the container by using the Shell script "run_local.sh":

```
./run_local.sh
```

You should see the output of your container in the terminal:

```
Executing entry point:                                                                                                                                                                                                                                                                                                        
---------------- BEGINNING OF CONTAINER EXECUTION ----------------------
Starting Monitoring Custom Metrics
Retrieving data from path: /opt/ml/processing/input/data
  Reading data from file: /opt/ml/processing/input/data
Finished retrieving data from path: /opt/ml/processing/input/data
Determining operation to run based on provided parameters ...
Determining monitor type ...
Monitor type detected based on 'analysis_type' environment variable
Operation type: OperationType.run_monitor
Monitor type: MonitorType.DATA_QUALITY
<class 'pandas.core.frame.DataFrame'>
...
```

The output files will be available in the "local_output" folder after the execution.


# Running the package in SageMaker

To use `MonitoringCustomMetrics` in a SageMaker Processing Job, you will need to:

- Configure AWS CLI.
- Containerize the code using Docker.
- Create an ECR Repo for MonitoringCustomMetrics in your AWS account.
- Create an IAM Role with Trust Relationship with SageMaker.
- Create an S3 bucket that will contain the input and output files.
- Start a SageMaker Processing Job.

### Configure AWS CLI

You will need to set up your AWS CLI. Choose the authentication method that best suits you: 
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html#getting-started-prereqs-keys

### Containerize the code using Docker

You can build the container using the following command:

```
docker build . --load
```

We need to identify the IMAGE_ID of our container. We can do so by running:

```
docker images
```

From the list, we should grab the most recent IMAGE_ID. We will use it in the next steps.

### Create an ECR Repo for MonitoringCustomMetrics in your AWS account

We need an ECR Repo where the container images will be uploaded.

```
aws ecr create-repository --repository-name <Repository Name> --region <AWS Region> --image-tag-mutability MUTABLE
```

Log in to ECR with AWS CLI:

```
aws ecr get-login-password --region <AWS Region> | docker login --username AWS --password-stdin <AWS Account ID>.dkr.ecr.<AWS Region>.amazonaws.com
```

Then we need to tag the image and push it to ECR. The "image tag" will be used to identify the container. You can use "MonitoringCustomMetrics" or any other name you prefer:

```
docker tag <Image Id> <AWS Account ID>.dkr.ecr.<AWS Region>.amazonaws.com/<Repository Name>:<Image Tag>

docker push <AWS Account ID>.dkr.ecr.<AWS Region>.amazonaws.com/Repository Name>:<Image Tag>
```

### Create an IAM Role with Trust Relationship with SageMaker

We need an IAM Role that has a trust relationship with SageMaker. We can create a trust policy file with this content:

trust-policy.json
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

Then we can create the role and attach the policy:

```
aws iam create-role --role-name <Role Name> --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess --role-name <Role Name>
```

### Create an S3 bucket that will contain the input and output files

You can create a new S3 bucket through AWS Console or through AWS CLI.

```
aws s3api create-bucket --bucket <S3 Bucket Name> --create-bucket-configuration LocationConstraint=<AWS Region>
```

You can now create folders inside the bucket, and upload the necessary files to each:
- input
- output
- baseline

Now we need to update the bucket policy, so that the IAM Role we created can read/write to the bucket:

bucket-policy.json
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::<AWS Account ID>:role/<Role Name>"
            },
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::<S3 Bucket Name>/*"
        }
    ]
}
```

Run the command to update the policy:

```
aws s3api put-bucket-policy --bucket <S3 Bucket Name> --policy file://bucket-policy.json
```

### Start a SageMaker Processing Job

Once all the required resources have been created in your AWS Account, you can now launch a Processing Job with the following command:

```
aws sagemaker create-processing-job \
	--processing-job-name <Name of the processing job> \
	--app-specification ImageUri="<ECR Image URI>",ContainerEntrypoint="python","./src/monitoring_custom_metrics/main.py" \
	--processing-resources 'ClusterConfig={InstanceCount=1,InstanceType="<Instance type to use>",VolumeSizeInGB=5}' \
	--role-arn <ARN of the IAM Role we created> \
	--environment analysis_type=DATA_QUALITY \
	--processing-inputs='[{"InputName": "dataInput", "S3Input": {"S3Uri": "<S3 Path to your input location>","LocalPath":"/opt/ml/processing/input/data","S3InputMode":"File", "S3DataType":"S3Prefix"}}]' \
	--processing-output-config 'Outputs=[{OutputName="report",S3Output={S3Uri="<S3 Path to your output location>",LocalPath="/opt/ml/processing/output",S3UploadMode="Continuous"}}]'
```

You should get a response from AWS CLI similar to:

```
{
    "ProcessingJobArn": "arn:aws:sagemaker:<AWS Region>:<AWS Account ID>:processing-job/<Name of the processing job>"
}
```

You can also see the SageMaker Processing Job in AWS Console. Once the job finishes, you will find the result files in the
output location you specified in the "processing-output-config" parameter.

# Available metrics

## Data Quality

|Metric name|Description|Data type|
|---|---|---|
|sum|Example metric that sums up an entire column's data|Numerical|
|email|Example metric to verify that a field is not an email|String|


## Model Quality

|Metric name|Description|Output data type| Parameters|
|---|---|---|---|
|brier_score_loss|	The Brier score measures the mean squared difference between the predicted probability and the actual outcome. Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html|Numerical|<ul><li>ground_truth_attribute: [required] str. Model target attribute.</li><li>probability_attribute: [required] str. Model inference attribute</li><li>threshold_override:[optional] float. Set constraint as baseline value + threshold_override.</li></ul>|
|gini|GINI is a model performance metric commonly used in Credit Science. It measures the ranking power of a model and it ranges from 0 to 1: 0 means no ranking power while 1 means perfect ranking power|Numerical|<ul><li>ground_truth_attribute: [required] str. Model target attribute.</li><li>probability_attribute: [required] str. Model inference attribute.</li><li>threshold_override:[optional] float. Set constraint as baseline value + threshold_override.</li></ul>|
|pr_auc|PR AUC is the area under precision-recall curve. Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html|Numerical|<ul><li>ground_truth_attribute: [required] str. Model target attribute.</li><li>probability_attribute: [required] str. Model inference attribute.</li><li>threshold_override:[optional] float. Set constraint as baseline value + threshold_override.</li></ul>|
|score_diff|Score difference measures the absolute/relative difference between predicted probability and the actual outcome.|Numerical|<ul><li>ground_truth_attribute: [required] str. Model target attribute.</li><li>probability_attribute: [required] str. Model inference attribute.</li><li>comparison_type: [optional] str. "absolute" to calculate absolute difference and "relative" to calculate relative difference. Default value is "absolute".</li><li>two_sided: [optional] bool. Default value is False:	<ul>		<li>two_sided = True will set the constraint and violation policy by the absolute value of the score difference to enable the detection of both under-prediction and over-prediction at the same time. The absolute value of score difference will be returned.</li>		<li>two_sided = False will set the constraint and violation policy by the original value of the score difference.</li>	</ul></li><li>comparison_operator: [optional] str. configure comparison_operator when two_sided is set as False. "GreaterThanThreshold" to detect over-prediction and "LessThanThreshold" to detect under-prediction.</li><li>threshold_override:[optional] float. Set constraint as baseline value + threshold_override.</li></ul>|



# How to implement additional metrics

Each metric is defined in its own class file. The file must be created in the right folder, based on the metric type:

- data_quality
  - numerical
  - string
- model_quality
  - binary_classification
  - multiclass_classification
  - regression

## Unit tests

Metrics must also have a unit test file in the "test" folder, following the same structure.

## Metric class conventions

- A metric must inherit from an Abstract Base Class (ABC) called "ModelQualityMetric".
- The class must include the following methods:
  - calculate_statistics.
  - suggest_constraints.
  - evaluate_constraints.
- At the end of the class, the file must expose a variable called "instance", which is an instance of the class itself.

Please refer to the existing metrics for additional details.