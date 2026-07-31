# Getting Started with Redis

This tutorial shows you how to install Redis and use it for the first time.

## Step 1: Install Redis

On Ubuntu, you can install Redis using apt:

```bash
sudo apt-get install redis-server
```

On macOS using Homebrew:

```bash
brew install redis
```

## Step 2: Start Redis

Start the Redis server:

```bash
redis-server
```

> **Tip:** Use redis-server --daemonize yes to run Redis in the background.

## Step 3: Connect with redis-cli

Connect to your Redis server using the CLI:

```bash
redis-cli
```

Try your first commands:

```redis
127.0.0.1:6379> SET mykey "Hello"
OK
127.0.0.1:6379> GET mykey
"Hello"
```
