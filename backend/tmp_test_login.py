from app.schemas import StudentLoginRequest
from app.services import school_portal_service

payload = StudentLoginRequest(enrolment_number="202002113200", password="aeli3200")
result = school_portal_service.authenticate_student(payload)
print(result)
