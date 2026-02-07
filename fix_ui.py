with open('game_ui_server.py', 'r') as f:
    content = f.read()
content = content.replace('import config', 'import config_server as config')
content = content.replace('server_name=127.0.0.1', 'server_name=0.0.0.0')
with open('game_ui_server.py', 'w') as f:
    f.write(content)
print('Fixed!')
