run-client:
	npm run dev --prefix ./client-side

run-server:
	python ./server_side/manage.py $(mode)

update-reqs:
	pip list --format=freeze > ./server_side/requirements.txt