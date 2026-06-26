"""
chunks = await store.search(query)
doc_ids = {c.document_id for c in chunks}
documents = await doc_store.get_many(doc_ids)
doc_by_id = {d.id: d for d in documents}
"""