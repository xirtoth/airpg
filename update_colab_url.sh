#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./update_colab_url.sh <gradio-url>"
    echo "Example: ./update_colab_url.sh https://abc123xyz.gradio.live"
    exit 1
fi

GRADIO_URL=$1
GRADIO_HOST=$(echo $GRADIO_URL | sed 's|https://||' | sed 's|http://||')

cat > /etc/nginx/sites-available/henkkaai.xyz << EOF
server {
    listen 80;
    server_name henkkaai.xyz www.henkkaai.xyz;
    
    location / {
        proxy_pass $GRADIO_URL;
        proxy_ssl_server_name on;
        proxy_set_header Host $GRADIO_HOST;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

nginx -t && systemctl reload nginx
echo "✅ Updated henkkaai.xyz to forward to $GRADIO_URL"
