FROM public.ecr.aws/lambda/python:3.12

ENV LD_LIBRARY_PATH="/lib64:/lib:/opt/amazon/lib:${LD_LIBRARY_PATH}"

# Copy over build directory
COPY . /opt/amazon

# Create input and output directories
RUN mkdir -p /opt/ml/processing/input/data
RUN mkdir -p /opt/ml/processing/output
RUN mkdir -p /opt/ml/processing/input/parameters
RUN mkdir -p /opt/ml/processing/baseline/constraints
RUN mkdir -p /opt/ml/processing/baseline/statistics
RUN mkdir -p /opt/ml/output

# Prepend our bin to path so that our executables are found when invoked by SageMaker.
ENV PATH=/opt/amazon/bin:/opt/amazon/sbin:${PATH}

# Add Python packages to path
ENV PYTHONPATH="/opt/amazon/lib/python3.8/site-packages:${PYTHONPATH}"

# Python environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONIOENCODING=UTF-8

##### Parameters for running locally should be put here: #####################################
ENV analysis_type=DATA_QUALITY
ENV baseline_statistics=/opt/ml/processing/baseline/statistics/community_statistics.json
ENV baseline_constraints=/opt/ml/processing/baseline/constraints/community_constraints.json
COPY test/resources/data_quality/input.csv /opt/ml/processing/input/data
COPY test/resources/data_quality/community_constraints.json /opt/ml/processing/baseline/constraints
COPY test/resources/data_quality/community_statistics.json /opt/ml/processing/baseline/statistics
##### End of Parameters for running locally ###########################################################################################

# Set working directory
WORKDIR /opt/amazon

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "./src/monitoring_custom_metrics/main.py"]