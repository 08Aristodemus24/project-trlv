run-client:
	npm run dev --prefix ./client-side

# make run-server mode=shell
# make run-server mode=runserver
# make run-server mode=makemigrations
# make run-server mode=migrate
run-server:
	python ./server_side/manage.py $(mode)

update-reqs:
	pip list --format=freeze > ./server_side/requirements.txt