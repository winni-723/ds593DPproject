from django.core.management.base import BaseCommand
from myapp.models import ITEM

class Command(BaseCommand):
    help = 'Fix spacing issues in professor names'

    def handle(self, *args, **options):
        # Get all items
        items = ITEM.objects.all()
        fixed_count = 0
        
        for item in items:
            # Clean up the professor name
            original_name = item.professor_name
            cleaned_name = ' '.join(original_name.split())  # This removes extra spaces
            
            if original_name != cleaned_name:
                item.professor_name = cleaned_name
                item.save()
                fixed_count += 1
                self.stdout.write(f"Fixed: '{original_name}' -> '{cleaned_name}'")
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully fixed {fixed_count} professor names')
        )
