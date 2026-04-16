# Usage
## Start Drupal
```
docker-compose -f drupal/docker-compose.yml up -d
```

## Log in
http://localhost:3470
username
: admin
password
: test

## Enable the required modules
http://localhost:3470/en/admin/modules
Install
- JSON:API

## Configure authorization

### None
For instances with public data and API.

### Access token
Access token can be used if you have [Key auth](https://www.drupal.org/project/key_auth) module or [Simple OAuth](https://www.drupal.org/project/simple_oauth). You can generate the token in Drupal administration interface.

### Username and password
Username and password can also be used, but they would also be stored in plain text in Learn2RAG configuration files.
