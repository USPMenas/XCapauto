FROM python:3.10-slim

# Instala pacotes essenciais, Chrome, ChromeDriver e cron
RUN apt-get update && apt-get install -y \
    wget unzip curl gnupg \
    chromium chromium-driver \
    cron \
    && apt-get clean

# Variáveis de ambiente para o Chrome headless
ENV CHROME_BIN="/usr/bin/chromium"
ENV PATH="$PATH:/usr/bin"

# Diretório de trabalho
WORKDIR /app

# Copia os arquivos
COPY requirements.txt .
COPY nasdaq_ndx_mail.py .
COPY run.sh .
COPY crontab /etc/cron.d/my-cron

# Instala dependências Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Permissões de execução e configuração do cron
RUN chmod +x run.sh
RUN chmod 0644 /etc/cron.d/my-cron
RUN crontab /etc/cron.d/my-cron

# Mantém o cron rodando em foreground no container
CMD ["cron", "-f"]
