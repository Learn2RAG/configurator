# Test Drupal instance
## Manual tests
### Start Drupal
```
docker compose -f drupal/docker-compose.yml up
```

### Details
base URL
: http://localhost:3470
username
: admin
password
: test
oauth client
: test_client
oauth secret
: test_secret

## Automated tests
The relevant tests automatically use this `docker compose` file, if docker is available.

# Data import from your instance
## Configure authorization
### None
For instances with public data and API.

### Access token
Access token can be used if you have [Key auth](https://www.drupal.org/project/key_auth) module or [Simple OAuth](https://www.drupal.org/project/simple_oauth). You can generate the token in Drupal administration interface.

### Username and password
Username and password can also be used, but they would also be stored in plain text in Learn2RAG configuration files.
