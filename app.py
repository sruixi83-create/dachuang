import ast
import os
import datetime  # 补充时间处理

import mysql.connector
from flask import Flask, request, render_template, redirect, url_for, session, flash
from mysql.connector import errorcode

# 确保Main.py存在且load_model/predict_image可正常调用
from Main import load_model, predict_image

# 创建Flask应用实例
app = Flask(__name__)
app.secret_key = 'a099670080289622e6de238568c8f659'

# 配置 MySQL 数据库（适配8.0+认证插件，补充端口/字符集）
config = {
    'user': 'root',
    'password': 'mysql',
    'host': 'localhost',
    'database': 'medical_platform',
    'port': 3306,  # 显式指定默认端口
    'auth_plugin': 'mysql_native_password',  # 解决8.0+认证兼容问题
    'charset': 'utf8mb4',  # 避免中文乱码
    'time_zone': '+8:00'  # 时区配置（可选，避免时间错乱）
}

# 定义上传文件的保存目录
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型（添加异常处理，避免启动失败）
try:
    model, _ = load_model()
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败：{str(e)}")
    model = None  # 兜底，避免后续调用报错


# 优化连接数据库的函数（返回conn+cursor，添加判空，统一关闭逻辑）
def get_db_connection():
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor(dictionary=True)  # 默认返回字典游标，更易用
        print("数据库连接成功")
        return conn, cursor
    except mysql.connector.Error as err:
        # 精准提示错误类型，方便排查
        error_msg = ""
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            error_msg = "MySQL访问被拒：用户名/密码错误，或无localhost权限"
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            error_msg = f"数据库不存在：medical_platform未创建"
        elif err.errno == errorcode.ER_CONN_REFUSED:
            error_msg = "MySQL服务未启动，或3306端口被拦截"
        else:
            error_msg = f"数据库连接失败：{str(err)}"
        print(f"❌ {error_msg}")
        flash(error_msg)  # 前端提示用户
        return None, None


# 安全关闭数据库连接（统一兜底）
def close_db_connection(conn, cursor):
    if cursor:
        try:
            cursor.close()
        except:
            pass
    if conn:
        try:
            conn.close()
        except:
            pass


# 主页路由
@app.route('/')
def index():
    return render_template('index.html')


# 登录页面路由（添加连接判空+异常处理）
@app.route('/login', methods=['GET', 'POST'])
def login():
    username = ''
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        # 1. 获取连接并判空
        conn, cursor = get_db_connection()
        if not conn or not cursor:
            return render_template('login.html', username=username)

        try:
            # 2. 执行SQL（参数化查询，防止注入）
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()

            if user and user['password'] == password:
                # 检查医生审核状态
                if user['role'] == 'doctor':
                    cursor.execute("SELECT is_approved FROM doctors WHERE user_id = %s", (user['id'],))
                    doctor_info = cursor.fetchone()
                    if not doctor_info or doctor_info['is_approved'] == 0:
                        flash('医生账号需管理员审核后方可登录')
                        return render_template('login.html', username=username)

                # 登录成功，写入session
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                flash('登录成功！')
                return redirect(url_for('index'))
            else:
                flash('用户名或密码错误！')
        except Exception as e:
            flash(f"登录失败：{str(e)}")
            print(f"登录SQL异常：{e}")
        finally:
            # 3. 无论成败，关闭连接
            close_db_connection(conn, cursor)

    return render_template('login.html', username=username)


# 注册页面路由（优化连接+异常处理+时间字段）
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        role = request.form.get('role', 'patient')

        # 空值校验
        if not username or not password:
            flash('用户名和密码不能为空！')
            return render_template('register.html')

        # 获取连接
        conn, cursor = get_db_connection()
        if not conn or not cursor:
            return render_template('register.html')

        try:
            # 检查用户名是否已存在
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                flash('用户名已存在！')
                return render_template('register.html')

            # 插入users表（补充created_at字段）
            cursor.execute(
                "INSERT INTO users (username, password, role, created_at) VALUES (%s, %s, %s, %s)",
                (username, password, role, datetime.datetime.now())
            )
            user_id = cursor.lastrowid

            # 插入对应角色表
            if role == 'patient':
                cursor.execute(
                    "INSERT INTO patients (user_id, name, created_at) VALUES (%s, %s, %s)",
                    (user_id, username, datetime.datetime.now())
                )
                flash('注册成功！您现在可以登录了。')
            elif role == 'doctor':
                cursor.execute(
                    "INSERT INTO doctors (user_id, name, specialty, is_approved, created_at) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, username, '', 0, datetime.datetime.now())
                )
                flash('注册成功！医生账号需管理员审核后方可登录。')

            conn.commit()  # 提交事务
            return redirect(url_for('login'))
        except Exception as e:
            conn.rollback()  # 失败回滚
            flash(f'注册失败：{str(e)}')
            print(f"注册SQL异常：{e}")
        finally:
            close_db_connection(conn, cursor)

    return render_template('register.html')


# 医生页面路由（优化连接+异常处理）
@app.route('/doctor')
def doctor():
    # 权限校验
    if 'user_id' not in session or session['role'] != 'doctor':
        flash('请以医生身份登录！')
        return redirect(url_for('login'))

    conn, cursor = get_db_connection()
    if not conn or not cursor:
        return redirect(url_for('login'))

    try:
        # 获取当前医生ID
        cursor.execute("SELECT id FROM doctors WHERE user_id = %s", (session['user_id'],))
        doctor_info = cursor.fetchone()
        if not doctor_info:
            flash('医生信息不存在！')
            return redirect(url_for('logout'))
        doctor_id = doctor_info['id']

        # 查询已绑定病人的检测记录
        query_bound = """
            SELECT d.*, p.name AS patient_name, d.image_path, d.result
            FROM detections d
            JOIN patients p ON d.patient_id = p.id
            JOIN doctor_patient_relations a ON d.patient_id = a.patient_id
            WHERE a.doctor_id = %s AND a.is_approved = 1
            ORDER BY d.patient_id, d.created_at DESC
        """
        cursor.execute(query_bound, (doctor_id,))
        bound_records = cursor.fetchall()

        # 查询未绑定但指定该医生的检测记录
        query_unbound = """
            SELECT d.*, p.name AS patient_name, d.image_path, d.result
            FROM detections d
            JOIN patients p ON d.patient_id = p.id
            WHERE d.doctor_id = %s
              AND NOT EXISTS (
                SELECT 1
                FROM doctor_patient_relations a
                WHERE a.patient_id = d.patient_id AND a.doctor_id = %s AND a.is_approved = 1
            )
            ORDER BY d.patient_id, d.created_at DESC
        """
        cursor.execute(query_unbound, (doctor_id, doctor_id))
        unbound_records = cursor.fetchall()

        # 查询绑定关系
        cursor.execute("SELECT * FROM doctor_patient_relations WHERE doctor_id = %s", (doctor_id,))
        assignments = cursor.fetchall()

        # 解析result字段
        def parse_result(record):
            try:
                parsed_result = ast.literal_eval(record['result'])
                formatted_result = [f"{label} (置信度: {confidence})" for label, confidence in parsed_result]
                record['formatted_result'] = formatted_result
            except Exception as e:
                record['formatted_result'] = ["解析失败"]
            return record

        bound_records = [parse_result(r) for r in bound_records]
        unbound_records = [parse_result(r) for r in unbound_records]

        # 按病人姓名分组
        from collections import defaultdict
        grouped_bound = defaultdict(list)
        for r in bound_records:
            grouped_bound[r['patient_name']].append(r)
        grouped_unbound = defaultdict(list)
        for r in unbound_records:
            grouped_unbound[r['patient_name']].append(r)

        return render_template('doctor.html',
                               username=session['username'],
                               grouped_bound=grouped_bound,
                               grouped_unbound=grouped_unbound,
                               doctor_id=doctor_id,
                               assignments=assignments)
    except Exception as e:
        flash(f"加载医生页面失败：{str(e)}")
        print(f"医生页面异常：{e}")
        return redirect(url_for('index'))
    finally:
        close_db_connection(conn, cursor)


# 请求绑定路由（优化连接+异常处理）
@app.route('/request_assign_patient', methods=['POST'])
def request_assign_patient():
    if 'user_id' not in session or session['role'] != 'doctor':
        flash('请以医生身份登录！')
        return redirect(url_for('login'))

    doctor_id = request.form.get('doctor_id')
    patient_id = request.form.get('patient_id')
    if not doctor_id or not patient_id:
        flash('参数错误！')
        return redirect(url_for('doctor'))

    conn, cursor = get_db_connection()
    if not conn or not cursor:
        return redirect(url_for('doctor'))

    try:
        # 检查是否已有绑定请求
        cursor.execute("""
            SELECT 1 FROM doctor_patient_relations 
            WHERE doctor_id = %s AND patient_id = %s
        """, (doctor_id, patient_id))
        if cursor.fetchone():
            flash("该病人已有绑定请求，审核通过前不可重复提交")
            return redirect(url_for('doctor'))

        # 插入绑定请求（补充requested_at字段）
        cursor.execute(
            "INSERT INTO doctor_patient_relations (doctor_id, patient_id, requested_at, is_approved) VALUES (%s, %s, %s, %s)",
            (doctor_id, patient_id, datetime.datetime.now(), 0)
        )
        conn.commit()
        flash("已提交绑定请求，请等待管理员审核")
    except mysql.connector.IntegrityError as e:
        conn.rollback()
        if "Duplicate entry" in str(e):
            flash("该绑定关系已存在")
        else:
            flash("提交绑定请求失败")
        print(f"绑定请求异常：{e}")
    except Exception as e:
        conn.rollback()
        flash(f"提交绑定请求失败：{str(e)}")
        print(f"绑定请求异常：{e}")
    finally:
        close_db_connection(conn, cursor)

    return redirect(url_for('doctor'))


# 病人页面路由（优化连接+模型兜底+异常处理）
@app.route('/patient', methods=['GET', 'POST'])
def patient():
    if 'user_id' not in session or session['role'] != 'patient':
        flash('请以病人身份登录！')
        return redirect(url_for('login'))

    # 获取医生列表（GET请求）
    conn, cursor = get_db_connection()
    if not conn or not cursor:
        return redirect(url_for('login'))

    try:
        cursor.execute("SELECT id, name, specialty FROM doctors WHERE is_approved = 1")
        doctors = cursor.fetchall()
    except Exception as e:
        flash(f"加载医生列表失败：{str(e)}")
        doctors = []
    finally:
        close_db_connection(conn, cursor)

    # 处理上传（POST请求）
    if request.method == 'POST':
        file = request.files.get('file')
        doctor_id = request.form.get('doctor_id')

        # 空值校验
        if not file or file.filename == '':
            flash("请选择文件上传！")
            return redirect(request.url)
        if not doctor_id:
            flash("请选择接诊医生！")
            return redirect(request.url)

        # 模型兜底校验
        if not model:
            flash("模型加载失败，无法进行预测！")
            return redirect(request.url)

        # 保存文件
        try:
            filepath_in_static = 'uploads/' + file.filename
            filepath = os.path.join('static', filepath_in_static)
            file.save(filepath)
        except Exception as e:
            flash(f"文件保存失败：{str(e)}")
            return redirect(request.url)

        # 图片预测（添加异常处理）
        try:
            result = predict_image(model, filepath)
            predicted_labels = [(label, f"{confidence:.2f}") for label, confidence in result]
        except Exception as e:
            flash(f"预测失败：{str(e)}")
            print(f"预测异常：{e}")
            return redirect(request.url)

        # 写入数据库
        conn, cursor = get_db_connection()
        if not conn or not cursor:
            return redirect(request.url)

        try:
            # 获取病人ID
            cursor.execute("SELECT id FROM patients WHERE user_id = %s", (session['user_id'],))
            patient = cursor.fetchone()
            if not patient:
                flash("您的病人信息不存在，请联系管理员！")
                return redirect(url_for('logout'))
            patient_id = patient['id']

            # 插入检测记录（补充created_at字段）
            cursor.execute(
                "INSERT INTO detections (patient_id, doctor_id, image_path, result, created_at) VALUES (%s, %s, %s, %s, %s)",
                (patient_id, doctor_id, filepath_in_static, str(predicted_labels), datetime.datetime.now())
            )
            conn.commit()
            flash("检测记录已保存！")
            return render_template('patient.html',
                                   filename=file.filename,
                                   predictions=predicted_labels,
                                   doctors=doctors)
        except Exception as e:
            conn.rollback()
            flash(f"保存检测记录失败：{str(e)}")
            print(f"检测记录异常：{e}")
        finally:
            close_db_connection(conn, cursor)

    return render_template('patient.html', doctors=doctors)


# 历史诊断记录页面路由（优化连接+异常处理）
@app.route('/history')
def history():
    if 'user_id' not in session or session['role'] != 'patient':
        flash('请以病人身份登录！')
        return redirect(url_for('login'))

    conn, cursor = get_db_connection()
    if not conn or not cursor:
        return redirect(url_for('login'))

    try:
        # 获取病人ID
        cursor.execute("SELECT id FROM patients WHERE user_id = %s", (session['user_id'],))
        patient_info = cursor.fetchone()
        if not patient_info:
            flash("您的病人信息不存在！")
            return redirect(url_for('profile'))
        patient_id = patient_info['id']

        # 查询检测记录
        query = """
            SELECT d.*, doc.name AS doctor_name
            FROM detections d
            LEFT JOIN doctors doc ON d.doctor_id = doc.id
            WHERE d.patient_id = %s
            ORDER BY d.created_at DESC
        """
        cursor.execute(query, (patient_id,))
        records = cursor.fetchall()

        # 解析result字段
        for record in records:
            try:
                parsed_result = ast.literal_eval(record['result'])
                formatted_result = [f"{label} (置信度: {confidence})" for label, confidence in parsed_result]
                record['formatted_result'] = formatted_result
            except Exception as e:
                record['formatted_result'] = ["解析失败"]

        return render_template('history.html', records=records)
    except Exception as e:
        flash(f"加载历史记录失败：{str(e)}")
        print(f"历史记录异常：{e}")
        return redirect(url_for('patient'))
    finally:
        close_db_connection(conn, cursor)


# 管理员页面路由（优化连接+异常处理）
@app.route('/admin')
def admin():
    if 'user_id' not in session or session['role'] != 'admin':
        flash('请以管理员身份登录！')
        return redirect(url_for('login'))

    conn, cursor = get_db_connection()
    if not conn or not cursor:
        return redirect(url_for('login'))

    try:
        # 查询病人/医生/绑定请求
        cursor.execute("SELECT * FROM patients")
        patients = cursor.fetchall()

        cursor.execute("SELECT * FROM doctors")
        doctors = cursor.fetchall()

        cursor.execute("""
            SELECT r.id, d.name AS doctor_name, p.name AS patient_name, r.requested_at
            FROM doctor_patient_relations r
            JOIN doctors d ON r.doctor_id = d.id
            JOIN patients p ON r.patient_id = p.id
            WHERE r.is_approved = 0
        """)
        requests = cursor.fetchall()

        return render_template('admin.html',
                               patients=patients,
                               doctors=doctors,
                               binding_requests=requests)
    except Exception as e:
        flash(f"加载管理员页面失败：{str(e)}")
        print(f"管理员页面异常：{e}")
        return redirect(url_for('index'))
    finally:
        close_db_connection(conn, cursor)


# 审核医生账号路由（优化连接+异常处理）
@app.route('/approve_doctor/<int:doctor_id>', methods=['POST'])
def approve_doctor(doctor_id):
    if 'user_id' not in session or session['role'] != 'admin':
        flash('请以管理员身份登录！')
        return redirect(url_for('login'))

    conn, cursor = get_db_connection()
    if not conn or not cursor:
        return redirect(url_for('admin'))

    try:
        cursor.execute(
            "UPDATE doctors SET is_approved = 1, approved_at = %s WHERE id = %s",
            (datetime.datetime.now(), doctor_id)
        )
        conn.commit()
        flash("医生账号已审核通过！")
    except Exception as e:
        conn.rollback()
        flash(f"审核失败：{str(e)}")
        print(f"审核医生异常：{e}")
    finally:
        close_db_connection(conn, cursor)

    return redirect(url_for('admin'))


# 批准绑定请求路由（优化连接+异常处理）
@app.route('/approve_binding_request/<int:request_id>', methods=['POST'])
def approve_binding_request(request_id):
    if 'user_id' not in session or session['role'] != 'admin':
        flash('请以管理员身份登录！')
        return redirect(url_for('login'))

    conn, cursor = get_db_connection()
    if not conn or not cursor:
        return redirect(url_for('admin'))

    try:
        # 获取请求信息
        cursor.execute("SELECT doctor_id, patient_id FROM doctor_patient_relations WHERE id = %s", (request_id,))
        req = cursor.fetchone()
        if not req:
            flash("绑定请求不存在！")
            return redirect(url_for('admin'))

        # 更新为已批准
        cursor.execute(
            "UPDATE doctor_patient_relations SET is_approved = 1, approved_at = %s WHERE id = %s",
            (datetime.datetime.now(), request_id)
        )
        conn.commit()
        flash("绑定请求已批准！")
    except Exception as e:
        conn.rollback()
        flash(f"批准失败：{str(e)}")
        print(f"批准绑定异常：{e}")
    finally:
        close_db_connection(conn, cursor)

    return redirect(url_for('admin'))


# 自定义模板过滤器：角色转中文
@app.template_filter('role_name')
def role_name(value):
    mapping = {
        'patient': '病人',
        'doctor': '医生',
        'admin': '管理员'
    }
    return mapping.get(value, value)


# 个人资料页面路由（优化连接+异常处理）
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('请先登录！')
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn, cursor = get_db_connection()
    if not conn or not cursor:
        return redirect(url_for('login'))

    try:
        # 获取用户基础信息
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            flash("用户不存在！")
            return redirect(url_for('logout'))

        # 获取角色扩展信息
        extra_info = {}
        if user['role'] == 'patient':
            cursor.execute("SELECT * FROM patients WHERE user_id = %s", (user_id,))
            extra_info = cursor.fetchone() or {}
        elif user['role'] == 'doctor':
            cursor.execute("SELECT * FROM doctors WHERE user_id = %s", (user_id,))
            extra_info = cursor.fetchone() or {}

        user.update(extra_info)
        return render_template('profile.html', user=user)
    except Exception as e:
        flash(f"加载个人资料失败：{str(e)}")
        print(f"个人资料异常：{e}")
        return redirect(url_for('index'))
    finally:
        close_db_connection(conn, cursor)


# 编辑个人资料页面路由（优化连接+异常处理）
@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        flash('请先登录！')
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn, cursor = get_db_connection()
    if not conn or not cursor:
        return redirect(url_for('login'))

    try:
        # 获取用户基础信息
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            flash("用户不存在！")
            return redirect(url_for('logout'))

        if request.method == 'POST':
            # 更新角色扩展信息
            if user['role'] == 'patient':
                name = request.form.get('name', '')
                gender = request.form.get('gender', '')
                age = request.form.get('age', None)
                if not name:
                    flash('姓名不能为空！')
                    return render_template('edit_profile.html', user=user)

                cursor.execute(
                    "UPDATE patients SET name = %s, gender = %s, age = %s, updated_at = %s WHERE user_id = %s",
                    (name, gender, age, datetime.datetime.now(), user_id)
                )
            elif user['role'] == 'doctor':
                name = request.form.get('name', '')
                specialty = request.form.get('specialty', '')
                if not name:
                    flash('姓名不能为空！')
                    return render_template('edit_profile.html', user=user)

                cursor.execute(
                    "UPDATE doctors SET name = %s, specialty = %s, updated_at = %s WHERE user_id = %s",
                    (name, specialty, datetime.datetime.now(), user_id)
                )

            conn.commit()
            flash("资料已更新！")
            return redirect(url_for('profile'))
        else:
            # GET请求：加载当前信息
            extra_info = {}
            if user['role'] == 'patient':
                cursor.execute("SELECT * FROM patients WHERE user_id = %s", (user_id,))
                extra_info = cursor.fetchone() or {}
            elif user['role'] == 'doctor':
                cursor.execute("SELECT * FROM doctors WHERE user_id = %s", (user_id,))
                extra_info = cursor.fetchone() or {}

            user.update(extra_info)
            return render_template('edit_profile.html', user=user)
    except Exception as e:
        conn.rollback()
        flash(f"更新资料失败：{str(e)}")
        print(f"编辑资料异常：{e}")
        return redirect(url_for('profile'))
    finally:
        close_db_connection(conn, cursor)


# 登出路由
@app.route('/logout')
def logout():
    session.clear()
    flash('已安全登出！')
    return redirect(url_for('login'))


# 启动Flask应用（开发环境）
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # 0.0.0.0允许局域网访问