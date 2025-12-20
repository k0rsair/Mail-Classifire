import os

from flask import Flask, request, jsonify
import numpy as np
import tempfile

app = Flask(__name__)

from classes import Category, EmbeddingService, MailClassyfire, MessageParser

# categories = [
#     Category(name="Финансы"),
#     Category(name="Юридические вопросы"),
#     Category(name="Техническая поддержка"),
#     Category(name="Бизнес"),
#     Category(name="Вакансия"),
#     Category(name="Промо")
#     # Либо любая категория, которая была введена пользователем
# ]

parser = MessageParser()

embedder = EmbeddingService()


mc = MailClassyfire()


@app.route('/api/predict', methods=['POST'])
def predict():
    # return jsonify(request.form)
    categories = [Category(cat) for cat in request.form.getlist('categories[]')]

    # файлы
    files = request.files.getlist('files[]')

    results = []

    for file in files:
        tmp_path = None

        try:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
                mail = parser.parse_email_file(tmp_path)
                results.append(mc.classify_email(mail['full_text'], categories,embedder,file.filename ))

        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

        finally:
            # 3️⃣ удаляем файл всегда
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return jsonify(results)

    return jsonify({
        'categories': categories,
        'files_count': len(files),
        'files_names': [f.filename for f in files]
    })
    # different_emails = []
    #
    # results = []
    #
    # for email in different_emails:
    #     result = mc.classify_email(
    #         email_text=email["full_text"],
    #         categories=categories,
    #         embedder=embedder
    #     )
    #
    #     results.append(result)
    #
    # return results
@app.route('/api/test', methods=['get'])
def test():
    return '123'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
