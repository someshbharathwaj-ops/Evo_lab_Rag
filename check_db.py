import psycopg, time

print('Verifying Supabase direct host...')
start = time.time()
conn = psycopg.connect(
    host='db.kpvjihhzruhykqyiwncs.supabase.co',
    port=5432,
    dbname='postgres',
    user='postgres',
    password='CoreLab@2026',
    sslmode='require',
    connect_timeout=15
)
print(f'Connected in {time.time()-start:.2f}s')

cur = conn.cursor()

# List all tables in public schema
cur.execute("""
    SELECT tablename
    FROM pg_tables 
    WHERE schemaname = 'public'
""")
tables = cur.fetchall()
print('\nTables in public schema:')
for t in tables:
    print(f'  {t[0]}')

# Count chunks
cur.execute("SELECT count(*) FROM public.chunks")
count = cur.fetchone()[0]
print(f'\nChunks stored: {count}')

# Sample a row
cur.execute("SELECT chunk_id, source, page, LEFT(text, 80) FROM public.chunks LIMIT 3")
rows = cur.fetchall()
print('\nSample rows:')
for r in rows:
    print(f'  [{r[2]}] {r[1]} — {r[3]}...')

conn.close()
print('\nAll good!')
