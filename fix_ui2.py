with open('game_ui_temp.py', 'r') as f:
    lines = f.readlines()

with open('game_ui_server.py', 'w') as f:
    for line in lines:
        if line.strip() == 'import config':
            f.write('import config_server as config\n')
        elif 'server_name=' in line and '127.0.0.1' in line:
            f.write(line.replace('127.0.0.1', '0.0.0.0'))
        else:
            f.write(line)
print('Fixed!')
