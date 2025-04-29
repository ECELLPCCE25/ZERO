import { betterAuth } from "better-auth";
import { Pool } from "pg";
 
export const auth = betterAuth({
    database: new Pool({
        connectionString: "postgresql://postgres.rmwpjmiihgrsawgtepez:TpkgD0LUGBLFAgA4@aws-0-ap-south-1.pooler.supabase.com:5432/postgres",
      }),
      emailAndPassword: {  
        enabled: true
    }
})