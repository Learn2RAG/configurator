$wgShowExceptionDetails = true;

# Require user login to read the content.
$wgGroupPermissions['*']['read'] = false;
$wgGroupPermissions['*']['edit'] = false;
$wgWhitelistRead = [
    'Special:UserLogin',
    'Special:UserLogout',
    'Special:PasswordReset',
    'Special:ChangeCredentials',
];
