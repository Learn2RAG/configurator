#!/bin/bash
set -eux

KEY_DIR="$PWD/private/oauth"
mkdir -p "$KEY_DIR"
chmod 750 "$KEY_DIR"

PRIVATE_KEY="${KEY_DIR}/private.key"
PUBLIC_KEY="${KEY_DIR}/public.key"

if [[ ! -s "$PRIVATE_KEY" ]]; then
  umask 077
  TEMP_PRIVATE="${PRIVATE_KEY}.tmp.$$"
  openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$TEMP_PRIVATE"
  mv "$TEMP_PRIVATE" "$PRIVATE_KEY"
fi

TEMP_PUBLIC="${PUBLIC_KEY}.tmp.$$"
openssl pkey -in "$PRIVATE_KEY" -pubout -out "$TEMP_PUBLIC"
mv "$TEMP_PUBLIC" "$PUBLIC_KEY"

chmod 640 "$PRIVATE_KEY"
chmod 644 "$PUBLIC_KEY"

drush config:set simple_oauth.settings private_key "$PRIVATE_KEY" -y
drush config:set simple_oauth.settings public_key "$PUBLIC_KEY" -y

drush php:eval '
$config = \Drupal::config("simple_oauth.settings");

foreach (["private_key", "public_key"] as $name) {
  $path = $config->get($name);

  if (!$path || !is_readable($path)) {
    throw new \RuntimeException("$name is not readable: $path");
  }
}
'
